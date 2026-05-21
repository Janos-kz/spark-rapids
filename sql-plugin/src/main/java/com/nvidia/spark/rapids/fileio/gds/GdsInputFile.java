/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.fileio.gds;

import ai.rapids.cudf.BaseDeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.CuFileBuffer;
import ai.rapids.cudf.CuFileReadHandle;
import ai.rapids.cudf.DeviceMemoryBuffer;
import com.nvidia.spark.rapids.jni.fileio.RapidsInputFile;
import com.nvidia.spark.rapids.jni.fileio.SeekableInputStream;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.File;
import java.io.IOException;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.Semaphore;

/**
 * Implementation of {@link RapidsInputFile} that supports GDS (GPUDirect Storage).
 * <br/>
 * This class provides the ability to read data directly from local NVMe/SSD storage
 * to GPU memory using the cuFile API, bypassing host memory for improved performance.
 * <br/>
 * It also supports fallback to standard Hadoop I/O when GDS is not available or
 * when the file doesn't meet GDS requirements.
 */
public class GdsInputFile implements RapidsInputFile {
    /** GDS requires 4KB alignment for optimal DMA performance */
    private static final long GDS_ALIGNMENT = 4096L;

    private static final int GDS_READ_PERMITS_COUNT =
            Integer.getInteger("rapids.gds.io.maxConcurrentReads", 8);
    private static final Semaphore GDS_READ_PERMITS =
            new Semaphore(GDS_READ_PERMITS_COUNT, true);

    private final Path filePath;
    private final FileSystem fs;
    private final File localFile;
    private final long fileSize;
    private final long minFileSize;
    private final boolean gdsEligible;

    /**
     * Create a GdsInputFile.
     *
     * @param filePath The Hadoop path to the file
     * @param conf Hadoop configuration
     * @param minFileSize Minimum file size for GDS eligibility (smaller files use Hadoop I/O)
     * @throws IOException if the file cannot be accessed
     */
    public GdsInputFile(Path filePath, Configuration conf, long minFileSize) throws IOException {
        Objects.requireNonNull(filePath, "filePath can't be null");
        Objects.requireNonNull(conf, "conf can't be null");

        this.filePath = filePath;
        this.fs = filePath.getFileSystem(conf);
        this.minFileSize = minFileSize;

        FileStatus status = fs.getFileStatus(filePath);
        this.fileSize = status.getLen();

        // Try to get the local file path
        String localPath = getLocalPath(filePath);
        this.localFile = localPath != null ? new File(localPath) : null;

        // Determine if GDS is eligible: must be a local file, exist, meet size threshold,
        // and cuFile must be available
        this.gdsEligible = localFile != null &&
                localFile.exists() &&
                fileSize >= minFileSize &&
                com.nvidia.spark.rapids.gds.GdsManager.isCuFileAvailable();
    }

    /**
     * Create a GdsInputFile with default minimum file size.
     */
    public GdsInputFile(Path filePath, Configuration conf) throws IOException {
        this(filePath, conf, 16L * 1024 * 1024); // Default 16MB
    }

    @Override
    public String path() {
        return filePath.toString();
    }

    @Override
    public long getLength() throws IOException {
        return fileSize;
    }

    @Override
    public OptionalLong getLastModificationTime() throws IOException {
        return OptionalLong.of(fs.getFileStatus(filePath).getModificationTime());
    }

    @Override
    public SeekableInputStream open() throws IOException {
        // For regular input stream, fall back to Hadoop I/O
        return new GdsInputStream(this);
    }

    /**
     * Check if this file is eligible for GDS operations.
     *
     * @return true if GDS can be used for this file
     */
    public boolean isGdsEligible() {
        return gdsEligible;
    }

    /**
     * Get the local File object for GDS operations.
     *
     * @return the local File, or null if not a local file
     */
    public File getLocalFile() {
        return localFile;
    }

    /**
     * Read data directly from the file to GPU memory using GDS.
     *
     * @param devBuffer The destination GPU memory buffer
     * @param offset The offset in the file to start reading
     * @param length The number of bytes to read
     * @throws IOException if the read fails
     */
    public void readToDeviceBuffer(BaseDeviceMemoryBuffer devBuffer, long offset, long length)
            throws IOException {
        if (!gdsEligible) {
            throw new IOException("File is not eligible for GDS: " + filePath);
        }
        if (offset + length > fileSize) {
            throw new IOException("Read beyond end of file: offset=" + offset +
                    ", length=" + length + ", fileSize=" + fileSize);
        }

        GDS_READ_PERMITS.acquireUninterruptibly();
        try {
            // cuFile JNI requires: offset 4KB-aligned, buffer size 4KB-aligned,
            // and bytes_read == buffer_size. It knows the logical file size.
            // Strategy:
            // 1. Align offset down to 4KB for DMA
            // 2. DMA read covers the 4KB-aligned portion of [offset, offset+length)
            // 3. Copy only the requested portion from DMA buffer to devBuffer
            // 4. The tail (< 4KB at file end) is read via host path

            long alignedOffset = (offset >> 12) << 12; // floor to 4KB
            long prefixSkip = offset - alignedOffset;
            long alignedFileEnd = (fileSize >> 12) << 12; // floor to 4KB

            // Calculate how much we can DMA: from alignedOffset to alignedFileEnd
            long dmaAvailFromAlignedOffset = alignedFileEnd - alignedOffset;

            // DMA read length: must be 4KB-aligned and within alignedFileEnd
            long dmaReadLen = Math.min(
                (prefixSkip + length + GDS_ALIGNMENT - 1) & ~(GDS_ALIGNMENT - 1),
                dmaAvailFromAlignedOffset & ~(GDS_ALIGNMENT - 1));
            if (dmaReadLen < GDS_ALIGNMENT && dmaAvailFromAlignedOffset >= GDS_ALIGNMENT) {
                dmaReadLen = GDS_ALIGNMENT;
            }

            // Phase 1: DMA read (4KB-aligned offset and size)
            long dmaCopiedToDest = 0;
            if (dmaReadLen >= GDS_ALIGNMENT) {
                try (CuFileBuffer cuFileBuf = CuFileBuffer.allocate(dmaReadLen, false);
                     Cuda.Stream myStream = new Cuda.Stream(true)) {
                    try (CuFileReadHandle readHandle =
                                 new CuFileReadHandle(localFile.getAbsolutePath())) {
                        readHandle.read(cuFileBuf, alignedOffset);
                    }
                    // Copy only the portion we need: from prefixSkip, up to min(length, dmaReadLen-prefixSkip)
                    long copyFromDma = Math.min(length, dmaReadLen - prefixSkip);
                    devBuffer.copyFromDeviceBufferAsync(
                            0, cuFileBuf, prefixSkip, copyFromDma, myStream);
                    myStream.sync();
                    dmaCopiedToDest = copyFromDma;
                }
            }

            // Phase 2: Host read for the tail (< 4KB at file end not covered by DMA)
            if (dmaCopiedToDest < length) {
                long hostOffset = offset + dmaCopiedToDest;
                long hostLength = length - dmaCopiedToDest;
                try (HostMemoryBuffer tailBuf = HostMemoryBuffer.allocate(hostLength);
                     java.io.RandomAccessFile raf = new java.io.RandomAccessFile(localFile, "r")) {
                    raf.seek(hostOffset);
                    byte[] tmp = new byte[(int) hostLength];
                    int bytesRead = raf.read(tmp, 0, (int) hostLength);
                    if (bytesRead > 0) {
                        tailBuf.setBytes(0, tmp, 0, bytesRead);
                        try (Cuda.Stream myStream = new Cuda.Stream(true)) {
                            devBuffer.copyFromHostBufferAsync(dmaCopiedToDest, tailBuf, 0, bytesRead, myStream);
                            myStream.sync();
                        }
                    }
                }
            }
        } catch (Exception e) {
            throw new IOException("GDS read failed for " + filePath, e);
        } finally {
            GDS_READ_PERMITS.release();
        }
    }


    /**
     * Read data directly from the file to a new GPU memory buffer using GDS.
     *
     * @param offset The offset in the file to start reading
     * @param length The number of bytes to read
     * @return A new DeviceMemoryBuffer containing the read data
     * @throws IOException if the read fails
     */
    public DeviceMemoryBuffer readToNewDeviceBuffer(long offset, long length)
            throws IOException {
        DeviceMemoryBuffer buffer = DeviceMemoryBuffer.allocate(length);
        try {
            readToDeviceBuffer(buffer, offset, length);
            return buffer;
        } catch (Exception e) {
            buffer.close();
            if (e instanceof IOException) {
                throw (IOException) e;
            }
            throw new IOException("GDS read failed for " + filePath, e);
        }
    }

    /**
     * Get the Hadoop FileSystem for this file.
     */
    FileSystem getFileSystem() {
        return fs;
    }

    /**
     * Extract the local file path from a Hadoop Path.
     * Returns null if the path is not a local file.
     */
    private String getLocalPath(Path path) {
        String scheme = path.toUri().getScheme();
        if (scheme == null || scheme.equals("file")) {
            // Local file
            String pathStr = path.toUri().getPath();
            if (pathStr == null || pathStr.isEmpty()) {
                pathStr = path.toString();
            }
            return pathStr;
        }
        return null;
    }
}
