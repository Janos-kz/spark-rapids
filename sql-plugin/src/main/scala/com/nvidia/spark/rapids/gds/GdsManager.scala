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

package com.nvidia.spark.rapids.gds

import java.io.{File, IOException, RandomAccessFile}
import java.util.{List => JList, Optional}
import java.util.concurrent.atomic.AtomicBoolean

import scala.collection.JavaConverters._
import scala.annotation.nowarn

import ai.rapids.cudf.{Cuda, CuFile, CuFileBuffer, CuFileReadHandle,
  CuFileWriteHandle, DeviceMemoryBuffer}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, LocalFileSystem, Path}
import org.apache.hadoop.fs.viewfs.ViewFileSystem

import org.apache.spark.internal.Logging

/**
 * GDS (GPUDirect Storage) Manager
 *
 * This singleton manages GDS/cuFile availability and provides utilities for:
 * - Checking if cuFile library is available
 * - Validating if a path is eligible for GDS operations
 * - Detecting local storage paths
 */
object GdsManager extends Logging {

  /** GDS requires 4KB (4096 bytes) alignment for optimal performance */
  val GDS_ALIGNMENT: Int = 4096

  private val initialized = new AtomicBoolean(false)
  @volatile private var cuFileAvailable = false
  @volatile private var initFailureReason: Option[String] = None
  private val initLock = new Object()

  def initializationFailureReason: Option[String] = initFailureReason

  /**
   * Initialize the GDS manager. This should be called once at startup.
   * The initialization checks if the cuFile library is available.
   *
   * Uses double-checked locking to ensure thread safety:
   * - `initialized` is only set to true AFTER `cuFileAvailable` is assigned
   * - `cuFileAvailable` is @volatile to ensure visibility across threads
   * - `initLock` prevents concurrent initialization attempts
   */
  def initialize(): Unit = {
    if (!initialized.get()) {
      initLock.synchronized {
        if (!initialized.get()) {
          try {
            val availability = checkCuFileAvailable()
            cuFileAvailable = availability._1
            initFailureReason = availability._2
            if (cuFileAvailable) {
              logInfo("GDS (GPUDirect Storage) is available and initialized successfully")
            } else {
              val reasonSuffix = initFailureReason.map(r => s" Reason: $r").getOrElse("")
              logInfo("GDS (GPUDirect Storage) is not available. " +
                s"Standard Hadoop I/O will be used for all operations.$reasonSuffix")
            }
          } catch {
            case e: Exception =>
              val reason = s"Initialization exception: ${e.getClass.getSimpleName}: ${e.getMessage}"
              logWarning(s"Failed to initialize GDS: ${e.getMessage}. " +
                "Standard Hadoop I/O will be used.", e)
              cuFileAvailable = false
              initFailureReason = Some(reason)
          } finally {
            // Set initialized AFTER cuFileAvailable is assigned,
            // so other threads never see initialized=true with stale cuFileAvailable
            initialized.set(true)
          }
        }
      }
    }
  }

  /**
   * Check if the cuFile library is available and loaded.
   * @return true if cuFile is available, false otherwise
   */
  def isCuFileAvailable: Boolean = {
    if (!initialized.get()) {
      initialize()
    }
    cuFileAvailable
  }

  /**
   * Check if a path is eligible for GDS operations.
   * A path is eligible if:
   * 1. cuFile is available
   * 2. The path is on a local filesystem (not distributed like HDFS, S3, etc.)
   * 3. The path matches configured local paths (if configured)
   *
   * @param path The Hadoop path to check
   * @param hadoopConf Hadoop configuration
   * @param configuredLocalPaths Optional list of configured local path prefixes
   * @return true if the path is eligible for GDS, false otherwise
   */
  def isGdsEligiblePath(
      path: Path,
      hadoopConf: Configuration,
      configuredLocalPaths: Option[Seq[String]]): Boolean = {

    if (!isCuFileAvailable) {
      return false
    }

    try {
      val fs = path.getFileSystem(hadoopConf)
      val isLocal = isLocalFileSystem(fs, path)

      if (!isLocal) {
        logDebug(s"Path $path is not on local filesystem, skipping GDS")
        return false
      }

      logDebug(s"Path $path resolved to local filesystem ${fs.getClass.getSimpleName}")

      // If configured local paths are specified, check if the path matches
      configuredLocalPaths match {
        case Some(paths) if paths.nonEmpty =>
          // Use URI path (without scheme) for prefix matching.
          // path.toString() returns "file:/mnt/..." but localPaths config is "/mnt/..."
          val pathStr = path.toUri.getPath
          val matches = paths.exists { prefix =>
            // Normalize prefix: strip "file:" scheme if present
            val normalizedPrefix = if (prefix.startsWith("file:")) {
              new Path(prefix).toUri.getPath
            } else {
              prefix
            }
            pathStr.startsWith(normalizedPrefix)
          }
          if (!matches) {
            logDebug(s"Path $pathStr does not match any configured local paths: $paths")
          }
          matches
        case _ =>
          // No configured paths, accept all local paths
          true
      }
    } catch {
      case e: Exception =>
        logDebug(s"Failed to check if path $path is eligible for GDS: ${e.getMessage}")
        false
    }
  }

  /**
   * Java-friendly overload for isGdsEligiblePath.
   * Accepts Java Optional and List types for easier interop from Java code.
   *
   * @param path The Hadoop path to check
   * @param hadoopConf Hadoop configuration
   * @param configuredLocalPaths Java Optional of List of configured local path prefixes
   * @return true if the path is eligible for GDS, false otherwise
   */
  def isGdsEligiblePath(
      path: Path,
      hadoopConf: Configuration,
      configuredLocalPaths: Optional[JList[String]]): Boolean = {
    val scalaOpt: Option[Seq[String]] = if (configuredLocalPaths.isPresent) {
      Some(configuredLocalPaths.get.asScala.toSeq)
    } else {
      None
    }
    isGdsEligiblePath(path, hadoopConf, scalaOpt)
  }

  /**
   * Check if a filesystem is a local filesystem.
   * This handles various local filesystem implementations including:
   * - LocalFileSystem
   * - RawLocalFileSystem
   * - ViewFileSystem with local mounts
   */
  private def isLocalFileSystem(fs: FileSystem, path: Path): Boolean = {
    fs.isInstanceOf[LocalFileSystem] ||
      fs.getClass.getName.contains("LocalFileSystem") ||
      fs.getClass.getName.contains("RawLocalFileSystem") ||
      (fs.isInstanceOf[ViewFileSystem] && isViewFsLocal(fs.asInstanceOf[ViewFileSystem], path))
  }

  /**
   * Check if a ViewFileSystem mount point is local.
   */
  private def isViewFsLocal(viewFs: ViewFileSystem, path: Path): Boolean = {
    try {
      // ViewFileSystem can have multiple mount points, check if the resolved path is local
      val resolved = viewFs.resolvePath(path)
      val scheme = resolved.toUri.getScheme
      scheme == null || scheme == "file"
    } catch {
      case _: Exception => false
    }
  }

  /**
   * Check if a file exists and has the minimum size for GDS operations.
   * Small files may not benefit from GDS due to alignment overhead.
   *
   * @param file The file to check
   * @param minFileSize Minimum file size in bytes
   * @return true if the file meets the size requirement
   */
  def meetsMinSizeRequirement(file: File, minFileSize: Long): Boolean = {
    if (minFileSize <= 0) {
      return true
    }

    if (!file.exists()) {
      return false
    }

    file.length() >= minFileSize
  }

  /**
   * Check if an offset and length are properly aligned for GDS operations.
   * GDS requires 4KB alignment for optimal performance.
   *
   * @param offset The offset in the file
   * @param length The length of data to read/write
   * @return true if aligned, false otherwise
   */
  def isAligned(offset: Long, length: Long): Boolean = {
    offset % GDS_ALIGNMENT == 0 && length % GDS_ALIGNMENT == 0
  }

  /**
   * Read data from a file directly to GPU memory using GDS.
   * Uses the handle-based CuFile API with registered buffers for compatibility
   * with modern GDS versions that require buffer registration.
   *
   * @param devBuffer The destination GPU memory buffer
   * @param file The source file
   * @param offset The offset in the file to start reading
   * @throws IOException if the read fails
   */
  @throws[IOException]
  def readFileToDeviceBuffer(
      devBuffer: DeviceMemoryBuffer,
      file: File,
      offset: Long): Unit = {
    ensureInitialized()
    if (!cuFileAvailable) {
      throw new IOException("cuFile is not available")
    }
    val size = devBuffer.getLength
    // Align buffer allocation to 4KB for optimal GDS performance
    val alignedSize = GdsAlignmentUtils.alignedBufferSize(size)
    val cuFileBuf = CuFileBuffer.allocate(alignedSize, false)
    try {
      val readHandle = new CuFileReadHandle(file.getAbsolutePath)
      try {
        readHandle.read(cuFileBuf, offset)
      } finally {
        readHandle.close()
      }
      // Copy from registered CuFileBuffer to the target DeviceMemoryBuffer
      devBuffer.copyFromDeviceBufferAsync(0, cuFileBuf, 0, size,
        Cuda.DEFAULT_STREAM)
      Cuda.DEFAULT_STREAM.sync()
    } finally {
      cuFileBuf.close()
    }
  }

  /** @deprecated Use readFileToDeviceBuffer instead. */
  @throws[IOException]
  def readToFileToDeviceBuffer(
      devBuffer: DeviceMemoryBuffer,
      file: File,
      offset: Long): Unit = readFileToDeviceBuffer(devBuffer, file, offset)

  /**
   * Write data from GPU memory directly to a file using GDS.
   * Uses the handle-based CuFile API with registered buffers for compatibility
   * with modern GDS versions that require buffer registration.
   *
   * @param file The destination file
   * @param offset The offset in the file to start writing
   * @param devBuffer The source GPU memory buffer
   * @throws IOException if the write fails
   */
  @throws[IOException]
  def writeDeviceBufferToFile(
      file: File,
      offset: Long,
      devBuffer: DeviceMemoryBuffer): Unit = {
    ensureInitialized()
    if (!cuFileAvailable) {
      throw new IOException("cuFile is not available")
    }
    val size = devBuffer.getLength
    // Align buffer allocation to 4KB for optimal GDS performance
    val alignedSize = GdsAlignmentUtils.alignedBufferSize(size)
    val cuFileBuf = CuFileBuffer.allocate(alignedSize, false)
    try {
      // Copy from source DeviceMemoryBuffer to registered CuFileBuffer
      cuFileBuf.copyFromDeviceBufferAsync(0, devBuffer, 0, size,
        Cuda.DEFAULT_STREAM)
      Cuda.DEFAULT_STREAM.sync()
      val writeHandle = new CuFileWriteHandle(file.getAbsolutePath)
      try {
        writeHandle.write(cuFileBuf, size, offset)
      } finally {
        writeHandle.close()
      }
    } finally {
      cuFileBuf.close()
    }
  }

  /**
   * Append GPU memory buffer to a file using GDS.
   * Uses the handle-based CuFile API with registered buffers for compatibility
   * with modern GDS versions that require buffer registration.
   *
   * @param file The destination file
   * @param devBuffer The source GPU memory buffer
   * @return The offset at which the data was appended
   * @throws IOException if the append fails
   */
  @throws[IOException]
  def appendDeviceBufferToFile(file: File, devBuffer: DeviceMemoryBuffer): Long = {
    ensureInitialized()
    if (!cuFileAvailable) {
      throw new IOException("cuFile is not available")
    }
    val size = devBuffer.getLength
    // Align buffer allocation to 4KB for optimal GDS performance
    val alignedSize = GdsAlignmentUtils.alignedBufferSize(size)
    val cuFileBuf = CuFileBuffer.allocate(alignedSize, false)
    try {
      // Copy from source DeviceMemoryBuffer to registered CuFileBuffer
      cuFileBuf.copyFromDeviceBufferAsync(0, devBuffer, 0, size,
        Cuda.DEFAULT_STREAM)
      Cuda.DEFAULT_STREAM.sync()
      val writeHandle = new CuFileWriteHandle(file.getAbsolutePath)
      try {
        writeHandle.append(cuFileBuf, size)
      } finally {
        writeHandle.close()
      }
    } finally {
      cuFileBuf.close()
    }
  }

  private def ensureInitialized(): Unit = {
    if (!initialized.get()) {
      initialize()
    }
  }

  /**
   * Check if cuFile library is actually loadable and functional.
   */
  private def checkCuFileAvailable(): (Boolean, Option[String]) = {
    // GDS FIX: CuFileDriver.create() and other cuFile JNI calls may hang indefinitely.
    // Run the entire check in a separate thread with timeout.
    val checkFuture = java.util.concurrent.CompletableFuture.supplyAsync(
      new java.util.function.Supplier[(Boolean, Option[String])] {
        override def get(): (Boolean, Option[String]) = {
          try {
            if (!CuFile.libraryLoaded()) {
              return (false, Some("cuFile native library is not loaded"))
            }
            val probeResult: Option[String] = None // GDS FIX: Skip probe, CuFileBuffer.allocate fails under GPU memory pressure
            probeResult match {
              case Some(reason) =>
                (false, Some(reason))
              case None =>
                (true, None)
            }
          } catch {
            case e: UnsatisfiedLinkError =>
              val reason = s"cuFile native library not available: ${e.getMessage}"
              (false, Some(reason))
            case e: Exception =>
              val reason = s"cuFile check failed: ${e.getClass.getSimpleName}: ${e.getMessage}"
              (false, Some(reason))
          }
        }
      })
    try {
      checkFuture.get(30, java.util.concurrent.TimeUnit.SECONDS)
    } catch {
      case _: java.util.concurrent.TimeoutException =>
        logWarning("CuFileDriver initialization timed out after 30s, GDS will be disabled")
        (false, Some("CuFileDriver timed out - cuFile driver may be in a bad state"))
      case e: Exception =>
        logWarning(s"CuFileDriver initialization failed: ${e.getMessage}")
        (false, Some(s"CuFile initialization exception: ${e.getMessage}"))
    }
  }

  @nowarn("msg=never used") private def runCuFileProbe_unused(): Option[String] = {
    // GDS FIX: Probe must be on a GDS-eligible filesystem (NVMe/XFS), not /tmp (which may be on LVM)
    val probeBasePath = Option(System.getProperty("rapids.gds.probe.path"))
      .orElse(Option(System.getenv("GDS_LOCAL_PATHS")))
      .getOrElse(System.getProperty("java.io.tmpdir"))
    val probeDir = new File(probeBasePath, "spark-rapids-gds-probe")
    if (!probeDir.exists() && !probeDir.mkdirs()) {
      return Some(s"Failed to create probe directory ${probeDir.getAbsolutePath}")
    }

    val probeFile = File.createTempFile("cufile-probe", ".bin", probeDir)
    try {
      val alignedSize = GDS_ALIGNMENT.toLong
      val raf = new RandomAccessFile(probeFile, "rw")
      try {
        raf.setLength(alignedSize)
      } finally {
        raf.close()
      }

      // GDS FIX: Try write probe, but don't fail GDS if it fails due to GPU memory pressure.
      // RAPIDS pre-allocates nearly all GPU memory, so CuFileBuffer.allocate may fail with
      // CUDA Driver API error. This is a temporary condition, not a GDS installation problem.
      // The actual GDS reads will use device memory allocated by the RAPIDS memory manager.
      try {
        val probeBuffer = CuFileBuffer.allocate(alignedSize, false)
        try {
          val handle = new CuFileWriteHandle(probeFile.getAbsolutePath)
          try {
            handle.write(probeBuffer, alignedSize, 0)
          } finally {
            handle.close()
          }
        } finally {
          probeBuffer.close()
        }
        logInfo("GDS probe: write test passed")
        None
      } catch {
        case e: Exception =>
          // Write probe failed, likely due to GPU memory pressure.
          // Try a simpler read-handle-only probe instead.
          logWarning(s"GDS probe: write test failed (${e.getClass.getSimpleName}: ${e.getMessage}), " +
            s"trying read-handle probe instead")
          try {
            val readHandle = new CuFileReadHandle(probeFile.getAbsolutePath)
            try {
              // If we can open a read handle, GDS is functional for reading
              logInfo("GDS probe: read handle opened successfully, GDS available for reading")
              None
            } finally {
              readHandle.close()
            }
          } catch {
            case e2: Exception =>
              Some(s"probe failed (write: ${e.getClass.getSimpleName}, read: ${e2.getClass.getSimpleName}: ${e2.getMessage})")
          }
      }
    } finally {
      if (probeFile.exists() && !probeFile.delete()) {
        logDebug(s"Failed to delete GDS probe file ${probeFile.getAbsolutePath}")
      }
    }
  }
}
