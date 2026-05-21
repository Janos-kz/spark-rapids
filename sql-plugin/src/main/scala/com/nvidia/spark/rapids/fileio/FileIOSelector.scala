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

package com.nvidia.spark.rapids.fileio

import java.util.{Optional, List => JList}
import scala.collection.JavaConverters._

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path

import org.apache.spark.internal.Logging

import com.nvidia.spark.rapids.RapidsConf
import com.nvidia.spark.rapids.fileio.gds.HybridFileIO
import com.nvidia.spark.rapids.gds.GdsManager
import com.nvidia.spark.rapids.jni.fileio.RapidsFileIO

/**
 * FileIO Selector
 *
 * This object provides a central point for selecting the appropriate FileIO
 * implementation based on configuration and path characteristics.
 */
object FileIOSelector extends Logging {

  /**
   * Select the appropriate FileIO implementation based on configuration and path.
   *
   * @param path The path to read/write (used to determine if GDS is applicable)
   * @param rapidsConf Rapids configuration
   * @param hadoopConf Hadoop configuration
   * @return A RapidsFileIO instance appropriate for the given configuration
   */
  def selectFileIO(
      path: Path,
      rapidsConf: RapidsConf,
      hadoopConf: Configuration): RapidsFileIO = {

    if (shouldUseGds(path, rapidsConf, hadoopConf)) {
      createHybridFileIO(rapidsConf, hadoopConf)
    } else {
      new hadoop.HadoopFileIO(hadoopConf)
    }
  }

  /**
   * Create a HybridFileIO instance from configuration.
   */
  def createHybridFileIO(
      rapidsConf: RapidsConf,
      hadoopConf: Configuration): HybridFileIO = {

    new HybridFileIO(
      hadoopConf,
      rapidsConf.isGdsIoInputEnabled,
      rapidsConf.isGdsIoOutputEnabled,
      rapidsConf.gdsIoMinFileSize,
      toJavaOptional(rapidsConf.gdsIoLocalPaths)
    )
  }

  /**
   * Convert Scala Option[Seq[String]] to Java Optional[List[String]].
   */
  private def toJavaOptional(opt: Option[Seq[String]]): Optional[JList[String]] = {
    opt.map(seq => Optional.of(seq.asJava)).getOrElse(Optional.empty())
  }

  /**
   * Create a FileIO instance without a specific path.
   * This returns a HybridFileIO if GDS is enabled, otherwise HadoopFileIO.
   */
  def createFileIO(
      rapidsConf: RapidsConf,
      hadoopConf: Configuration): RapidsFileIO = {

    if (rapidsConf.isGdsIoEnabled && GdsManager.isCuFileAvailable) {
      createHybridFileIO(rapidsConf, hadoopConf)
    } else {
      new hadoop.HadoopFileIO(hadoopConf)
    }
  }

  /**
   * Create a FileIO instance from pre-extracted GDS config values.
   * This variant is safe to use in @transient lazy val contexts where
   * RapidsConf may be null after deserialization.
   */
  def createFileIO(
      isGdsEnabled: Boolean,
      isGdsInputEnabled: Boolean,
      isGdsOutputEnabled: Boolean,
      gdsMinFileSize: Long,
      gdsLocalPaths: Option[Seq[String]],
      hadoopConf: Configuration): RapidsFileIO = {

    if (isGdsEnabled && GdsManager.isCuFileAvailable) {
      val initFailureSuffix = GdsManager.initializationFailureReason
        .map(reason => s", reason=$reason").getOrElse("")
      logInfo("FileIOSelector: Creating HybridFileIO (GDS enabled, cuFile available)" +
        s", gdsInput=$isGdsInputEnabled, gdsOutput=$isGdsOutputEnabled" +
        s", minFileSize=$gdsMinFileSize, localPaths=$gdsLocalPaths$initFailureSuffix")
      new HybridFileIO(
        hadoopConf,
        isGdsInputEnabled,
        isGdsOutputEnabled,
        gdsMinFileSize,
        toJavaOptional(gdsLocalPaths))
    } else {
      val initFailureSuffix = GdsManager.initializationFailureReason
        .map(reason => s", reason=$reason").getOrElse("")
      logInfo(s"FileIOSelector: Creating HadoopFileIO (GDS not used:" +
        s" gdsEnabled=$isGdsEnabled, cuFileAvailable=${if (isGdsEnabled) GdsManager.isCuFileAvailable else false}$initFailureSuffix)")
      new hadoop.HadoopFileIO(hadoopConf)
    }
  }

  /**
   * Create a FileIO instance from Hadoop configuration (without RapidsConf).
   * This is used in write paths where RapidsConf is not directly accessible.
   *
   * The GDS settings are read from Hadoop configuration keys that are set
   * from RapidsConf during job planning.
   */
  def createFileIOFromHadoopConf(hadoopConf: Configuration): RapidsFileIO = {
    // Read GDS settings from Hadoop configuration
    val gdsEnabled = hadoopConf.getBoolean(
      "spark.rapids.gds.io.enabled", false)
    val gdsInputEnabled = hadoopConf.getBoolean(
      "spark.rapids.gds.io.input.enabled", true)
    val gdsOutputEnabled = hadoopConf.getBoolean(
      "spark.rapids.gds.io.output.enabled", true)
    val gdsMinFileSize = hadoopConf.getLong(
      "spark.rapids.gds.io.minFileSize", 16L * 1024 * 1024) // Default 16MB, matching RapidsConf

    // Read local paths configuration
    val localPathsStr = hadoopConf.get("spark.rapids.gds.io.localPaths", "")
    val gdsLocalPaths: Option[Seq[String]] = if (localPathsStr.isEmpty) {
      None
    } else {
      Some(localPathsStr.split(",").map(_.trim).filter(_.nonEmpty).toSeq)
    }

    if (gdsEnabled && GdsManager.isCuFileAvailable) {
      val initFailureSuffix = GdsManager.initializationFailureReason
        .map(reason => s", reason=$reason").getOrElse("")
      logInfo(s"FileIOSelector (fromHadoopConf): Creating HybridFileIO" +
        s", gdsInput=$gdsInputEnabled, gdsOutput=$gdsOutputEnabled" +
        s", minFileSize=$gdsMinFileSize, localPaths=$gdsLocalPaths$initFailureSuffix")
      new HybridFileIO(
        hadoopConf,
        gdsInputEnabled,
        gdsOutputEnabled,
        gdsMinFileSize,
        toJavaOptional(gdsLocalPaths))
    } else {
      val initFailureSuffix = GdsManager.initializationFailureReason
        .map(reason => s", reason=$reason").getOrElse("")
      logInfo(s"FileIOSelector (fromHadoopConf): Creating HadoopFileIO" +
        s" (gdsEnabled=$gdsEnabled, cuFileAvailable=${if (gdsEnabled) GdsManager.isCuFileAvailable else false}$initFailureSuffix)")
      new hadoop.HadoopFileIO(hadoopConf)
    }
  }

  /**
   * Determine if GDS should be used for a given path and configuration.
   */
  def shouldUseGds(
      path: Path,
      rapidsConf: RapidsConf,
      hadoopConf: Configuration): Boolean = {

    // Check if GDS is enabled in configuration
    if (!rapidsConf.isGdsIoEnabled) {
      return false
    }

    // Check if cuFile library is available
    if (!GdsManager.isCuFileAvailable) {
      return false
    }

    // Check if the path is eligible for GDS
    GdsManager.isGdsEligiblePath(path, hadoopConf, rapidsConf.gdsIoLocalPaths)
  }

  /**
   * Check if GDS should be used for input operations.
   */
  def shouldUseGdsForInput(
      path: Path,
      rapidsConf: RapidsConf,
      hadoopConf: Configuration): Boolean = {

    rapidsConf.isGdsIoInputEnabled && shouldUseGds(path, rapidsConf, hadoopConf)
  }

  /**
   * Check if GDS should be used for output operations.
   */
  def shouldUseGdsForOutput(
      path: Path,
      rapidsConf: RapidsConf,
      hadoopConf: Configuration): Boolean = {

    rapidsConf.isGdsIoOutputEnabled && shouldUseGds(path, rapidsConf, hadoopConf)
  }
}
