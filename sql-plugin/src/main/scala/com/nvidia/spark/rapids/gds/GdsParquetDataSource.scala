package com.nvidia.spark.rapids.gds

import java.io.IOException

import ai.rapids.cudf._
import com.nvidia.spark.rapids.fileio.gds.GdsInputFile

class GdsParquetDataSource(
    gdsInputFile: GdsInputFile,
    fileSize: Long
) extends DataSource {

  override def size(): Long = fileSize

  override def hostRead(offset: Long, length: Long): HostMemoryBuffer = {
    val hmb = HostMemoryBuffer.allocate(length)
    try {
      val in = gdsInputFile.open()
      try {
        in.seek(offset)
        val tmp = new Array[Byte](8192)
        var totalRead = 0L
        while (totalRead < length) {
          val toRead = math.min(tmp.length, (length - totalRead).toInt)
          val read = in.read(tmp, 0, toRead)
          if (read < 0) throw new IOException("EOF in GDS DataSource")
          hmb.setBytes(totalRead, tmp, 0, read)
          totalRead += read
        }
      } finally { in.close() }
      hmb
    } catch { case e: Exception => hmb.close(); throw e }
  }

  override def hostRead(offset: Long, buffer: HostMemoryBuffer): Long = {
    val length = buffer.getLength
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
      totalRead
    } finally { in.close() }
  }

  override def supportsDeviceRead(): Boolean = false  // TEMP DIAGNOSTIC

  override def deviceRead(offset: Long, buffer: DeviceMemoryBuffer,
      stream: Cuda.Stream): Long = {
    val length = buffer.getLength
    // TEMP DIAGNOSTIC: Use host read + H2D copy to verify correctness
    // This rules out CuFileBuffer/CuFileReadHandle issues
    val hmb = hostRead(offset, length)
    try {
      buffer.copyFromHostBufferAsync(0, hmb, 0, length, stream)
      stream.sync()
    } finally {
      hmb.close()
    }
    length
  }

  override def close(): Unit = {}
}
