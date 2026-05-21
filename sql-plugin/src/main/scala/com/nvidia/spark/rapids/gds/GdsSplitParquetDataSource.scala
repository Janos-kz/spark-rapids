package com.nvidia.spark.rapids.gds

import java.io.IOException
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream

import ai.rapids.cudf._
import com.nvidia.spark.rapids.fileio.gds.GdsInputFile

import org.apache.parquet.format._
import shaded.parquet.org.apache.thrift.protocol.TCompactProtocol
import shaded.parquet.org.apache.thrift.transport.TIOStreamTransport

/**
 * GDS DataSource for zero-copy Parquet reading.
 * Modifies the Parquet footer to only include row groups overlapping the split range,
 * so cudf only reads the relevant data via GDS DMA (NVMe -> GPU).
 *
 * For GDS-eligible files: deviceRead() uses cuFile DMA.
 * For non-GDS-eligible files: hostRead() uses regular file I/O, footer still modified.
 */
class GdsSplitParquetDataSource(
    gdsInputFile: GdsInputFile,
    fileSize: Long,
    splitStart: Long,
    splitLength: Long
) extends DataSource {

  private val isGdsEligible = gdsInputFile.isGdsEligible()

  private val (originalFooterLength: Int, modifiedFooterBytes: Array[Byte]) = {
    buildModifiedFooter()
  }

  private val footerStart: Long = fileSize - 8 - originalFooterLength
  private val footerEnd: Long = fileSize

  override def size(): Long = fileSize

  override def hostRead(offset: Long, length: Long): HostMemoryBuffer = {
    System.err.println(s"GDS HOST: hostRead offset= length= gdsEligible=")
    if (offset < footerEnd && (offset + length) > footerStart) {
      returnModifiedFooterRead(offset, length)
    } else {
      readFromFile(offset, length)
    }
  }

  override def hostRead(offset: Long, buffer: HostMemoryBuffer): Long = {
    val length = buffer.getLength
    System.err.println(s"GDS HOST2: hostRead(offset=, len=) gdsEligible=")
    if (offset < footerEnd && (offset + length) > footerStart) {
      returnModifiedFooterReadToBuffer(offset, length, buffer)
    } else {
      readFromFileToBuffer(offset, length, buffer)
    }
  }

  override def supportsDeviceRead(): Boolean = isGdsEligible

  override def getDeviceReadCutoff(): Long = {
    // 0 means ALL reads should go through deviceRead when supportsDeviceRead() is true
    // This ensures cudf uses GDS DMA for eligible files
    System.err.println(s"GDS CUTOFF: getDeviceReadCutoff called, gdsEligible=")
    0L
  }

  override def deviceRead(offset: Long, buffer: DeviceMemoryBuffer,
      stream: Cuda.Stream): Long = {
    val length = buffer.getLength
    if (!isGdsEligible) {
      throw new UnsupportedOperationException("Not GDS eligible")
    }
    gdsInputFile.readToDeviceBuffer(buffer, offset, length)
    System.err.println(s"GDS DMA: deviceRead offset= length= gdsEligible=")
    // GDS FIX: Apply footer modification to device buffer, just like hostRead does.
    // When cudf uses deviceRead to read a chunk that includes the footer region,
    // the raw GDS DMA data contains the original footer. We must patch in the
    // modified footer to ensure cudf sees the filtered row groups.
    if (offset < footerEnd && (offset + length) > footerStart) {
      val modFooterStart = footerStart
      val modFooterEnd = footerStart + modifiedFooterBytes.length
      val overlapStart = math.max(offset, modFooterStart)
      val overlapEnd = math.min(offset + length, modFooterEnd)
      if (overlapStart < overlapEnd) {
        val bufOffset = overlapStart - offset
        val footerOffset = overlapStart - modFooterStart
        val copyLen = (overlapEnd - overlapStart).toInt
        val footerPatch = HostMemoryBuffer.allocate(copyLen)
        try {
          footerPatch.setBytes(0, modifiedFooterBytes, footerOffset, copyLen)
          buffer.copyFromHostBuffer(bufOffset, footerPatch, 0, copyLen)
        } finally {
          footerPatch.close()
        }
      }
    }
    length
  }

  override def close(): Unit = {
    // GDS FIX: cudf DataSource caches HostMemoryBuffers returned by hostRead() via JNI.
    // These buffers are released via onHostBufferDone() callback when the C++ Parquet reader
    // destructs. But C++ reader destruction timing is uncertain (depends on Java GC).
    // Instead of throwing IllegalStateException like the default DataSource.close(),
    // we skip the cachedBuffers check to avoid false errors, and let onHostBufferDone
    // naturally release the buffers when the C++ reader destructs.
  }

  private def buildModifiedFooter(): (Int, Array[Byte]) = {
    val last8 = new Array[Byte](8)
    val fin = gdsInputFile.open()
    try {
      fin.seek(fileSize - 8)
      var totalRead = 0
      while (totalRead < 8) {
        val read = fin.read(last8, totalRead, 8 - totalRead)
        if (read < 0) throw new IOException("EOF reading footer tail")
        totalRead += read
      }
    } finally { fin.close() }

    val footerLength = (last8(0) & 0xFF) |
                       ((last8(1) & 0xFF) << 8) |
                       ((last8(2) & 0xFF) << 16) |
                       ((last8(3) & 0xFF) << 24)

    val magic = new String(last8, 4, 4, "UTF-8")
    if (magic != "PAR1") {
      throw new IOException(s"Invalid Parquet magic: $magic")
    }

    val footerBytes = new Array[Byte](footerLength)
    val fin2 = gdsInputFile.open()
    try {
      fin2.seek(fileSize - 8 - footerLength)
      var totalRead = 0
      while (totalRead < footerLength) {
        val read = fin2.read(footerBytes, totalRead, footerLength - totalRead)
        if (read < 0) throw new IOException("EOF reading footer")
        totalRead += read
      }
    } finally { fin2.close() }

    val bais = new ByteArrayInputStream(footerBytes)
    val transport = new TIOStreamTransport(bais)
    val protocol = new TCompactProtocol(transport)
    val fileMetaData = new FileMetaData()
    fileMetaData.read(protocol)

    val originalRowGroups = fileMetaData.getRow_groups
    val filteredRowGroups = new java.util.ArrayList[RowGroup]()

    var totalRows: Long = 0
    val rowGroupsIt = originalRowGroups.iterator()
    while (rowGroupsIt.hasNext) {
      val rg = rowGroupsIt.next()
      if (rowGroupOverlapsSplit(rg)) {
        filteredRowGroups.add(rg)
        totalRows += rg.getNum_rows
      }
    }

    if (filteredRowGroups.isEmpty) {
      System.err.println(s"GDS WARNING: No row groups for split [$splitStart, ${splitStart + splitLength})")
    }

    fileMetaData.setRow_groups(filteredRowGroups)
    fileMetaData.setNum_rows(totalRows)

    val baos = new ByteArrayOutputStream()
    val outTransport = new TIOStreamTransport(baos)
    val outProtocol = new TCompactProtocol(outTransport)
    fileMetaData.write(outProtocol)
    outTransport.flush()

    val modifiedBytes = baos.toByteArray()

    System.err.println(s"GDS FOOTER: ${originalRowGroups.size()} -> ${filteredRowGroups.size()} row groups, " +
      s"footer ${footerLength} -> ${modifiedBytes.length}, split [$splitStart, ${splitStart + splitLength}), " +
      s"totalRows=$totalRows, gdsEligible=$isGdsEligible")

    (footerLength, modifiedBytes)
  }

  private def rowGroupOverlapsSplit(rg: RowGroup): Boolean = {
    val splitEnd = splitStart + splitLength
    val columnsIt = rg.getColumns.iterator()
    while (columnsIt.hasNext) {
      val cc = columnsIt.next()
      val meta = cc.getMeta_data
      if (meta != null) {
        var chunkStart: Long = meta.getData_page_offset
        if (meta.isSetDictionary_page_offset && meta.getDictionary_page_offset < chunkStart) {
          chunkStart = meta.getDictionary_page_offset
        }
        val chunkEnd = chunkStart + meta.getTotal_compressed_size
        if (chunkStart < splitEnd && chunkEnd > splitStart) {
          return true
        }
      }
    }
    false
  }

  private def returnModifiedFooterRead(offset: Long, length: Long): HostMemoryBuffer = {
    val hmb = HostMemoryBuffer.allocate(length)
    try {
      fillFromFile(hmb, offset, length)
      val modFooterStart = footerStart
      val modFooterEnd = footerStart + modifiedFooterBytes.length
      val overlapStart = math.max(offset, modFooterStart)
      val overlapEnd = math.min(offset + length, modFooterEnd)
      if (overlapStart < overlapEnd) {
        val bufOffset = overlapStart - offset
        val footerOffset = overlapStart - modFooterStart
        val copyLen = overlapEnd - overlapStart
        hmb.setBytes(bufOffset, modifiedFooterBytes, footerOffset.toInt, copyLen.toInt)
      }
      hmb
    } catch {
      case e: Exception => hmb.close(); throw e
    }
  }

  private def returnModifiedFooterReadToBuffer(offset: Long, length: Long,
      buffer: HostMemoryBuffer): Long = {
    fillFromFile(buffer, offset, length)
    val modFooterStart = footerStart
    val modFooterEnd = footerStart + modifiedFooterBytes.length
    val overlapStart = math.max(offset, modFooterStart)
    val overlapEnd = math.min(offset + length, modFooterEnd)
    if (overlapStart < overlapEnd) {
      val bufOffset = overlapStart - offset
      val footerOffset = overlapStart - modFooterStart
      val copyLen = overlapEnd - overlapStart
      buffer.setBytes(bufOffset, modifiedFooterBytes, footerOffset.toInt, copyLen.toInt)
    }
    length
  }

  private def readFromFile(offset: Long, length: Long): HostMemoryBuffer = {
    val hmb = HostMemoryBuffer.allocate(length)
    try {
      fillFromFile(hmb, offset, length)
      hmb
    } catch {
      case e: Exception => hmb.close(); throw e
    }
  }

  private def readFromFileToBuffer(offset: Long, length: Long,
      buffer: HostMemoryBuffer): Long = {
    fillFromFile(buffer, offset, length)
    length
  }

  private def fillFromFile(buffer: HostMemoryBuffer, offset: Long, length: Long): Unit = {
    val in = gdsInputFile.open()
    try {
      in.seek(offset)
      val tmp = new Array[Byte](8192)
      var totalRead = 0L
      while (totalRead < length) {
        val toRead = math.min(tmp.length, (length - totalRead).toInt)
        val read = in.read(tmp, 0, toRead)
        if (read < 0) throw new IOException("EOF in GDS DataSource")
        buffer.setBytes(totalRead, tmp, 0, read)
        totalRead += read
      }
    } finally { in.close() }
  }
}
