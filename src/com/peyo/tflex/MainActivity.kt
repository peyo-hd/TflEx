package com.peyo.tflex

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.SystemClock
import android.view.View
import com.peyo.tflex.databinding.MainBinding
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.util.*
import java.util.AbstractMap.SimpleEntry
import kotlin.collections.ArrayList
import kotlin.concurrent.thread

class MainActivity: Activity() {
    companion object {
        private const val TAG = "TFLEx01"
        private val images = arrayOf("test_image.jpg", "test_image1.jpg",
                "test_image2.jpg", "test_image3.jpg", "test_image4.jpg",
                "test_image5.jpg", "test_image.jpg", "test_image1.jpg",
                "test_image2.jpg", "test_image3.jpg", "test_image4.jpg",
                "test_image5.jpg", "test_image.jpg", "test_image1.jpg",
                "test_image2.jpg", "test_image3.jpg", "test_image4.jpg",
                "test_image5.jpg", "test_image.jpg", "test_image1.jpg",
                "test_image2.jpg", "test_image3.jpg", "test_image4.jpg",
                "test_image5.jpg")
    }

    private lateinit var binding: MainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = MainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        loadLabels()
        tfliteModel = FileUtil.loadMappedFile(this, "mobilenet_v1_1_0_224_float.tflite")
    }

    override fun onDestroy() {
        super.onDestroy()
        tflite.close()
    }

    fun onComputeClick(v: View) {
        thread {
            val options = Interpreter.Options()
            if (binding.nnapiToggle.isChecked) {
                options.addDelegate(NnApiDelegate())
            } else {
                options.setNumThreads(1)
            }
            tflite = Interpreter(tfliteModel, options)

            inferenceTime = 0
            firstFrame = true
            for(image in images) {
                convertBitmapToByteBuffer(getBitmap(image))

                startTime = SystemClock.uptimeMillis()
                tflite.run(imgData, outputs)
                printLabels()

                Thread.sleep(500)
            }

            runOnUiThread {
                binding.textView1.text = "Summary: \n\t Average Inference time (ms): " +
                        "${inferenceTime / (images.size - 1)}"
                binding.textView2.text = ""
            }
            tflite.close()
        }
     }

    private fun printLabels() {
        val runtime = SystemClock.uptimeMillis() - startTime

        for (i in 0 until getNumLabels()) {
            sortedLabels.add(SimpleEntry(labelList[i], outputs[0][i]))
            if (sortedLabels.size > RESULTS_TO_SHOW) {
                sortedLabels.poll()
            }
        }

        var text = ""
        for (i in 0 until sortedLabels.size) {
            val label = sortedLabels.poll()
            text = String.format("\n  %s: %f", label.key, label.value) + text
        }
        text = "Result:" + text

        runOnUiThread {
            binding.textView1.text = "Inference time (ms): " + runtime
            if (firstFrame) {
                firstFrame = false
            } else {
                inferenceTime += runtime
            }

            binding.textView2.text = text
        }
    }

    private var startTime: Long = 0
    private var inferenceTime : Long = 0
    private var firstFrame : Boolean = true

    private lateinit var tfliteModel: MappedByteBuffer
    private var imgData: ByteBuffer? = null
    private val intValues = IntArray(224 * 224)
    private val IMAGE_MEAN = 128.0f
    private val IMAGE_STD = 128.0f

    private fun convertBitmapToByteBuffer(bitmap: Bitmap) {
        if (imgData == null) {
            imgData = ByteBuffer.allocateDirect(
                    1 * 224 * 224 * 3 * 4)
            imgData!!.order(ByteOrder.nativeOrder())
        }
        imgData!!.rewind()
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        var pixel = 0
        for (i in 0 until 224) {
            for (j in 0 until 224) {
                val v: Int = intValues.get(pixel++)
                imgData!!.putFloat(((v shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                imgData!!.putFloat(((v shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                imgData!!.putFloat(((v and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
            }
        }
    }

    private val RESULTS_TO_SHOW = 3

    private val sortedLabels = PriorityQueue<Map.Entry<String, Float>>(RESULTS_TO_SHOW)
        { o1, o2 -> o1.value.compareTo(o2.value) }

    private lateinit var tflite: Interpreter
    private var labelList = ArrayList<String>()
    private lateinit var outputs: Array<FloatArray>

    private fun loadLabels() {
        val reader = BufferedReader(InputStreamReader(assets.open("labels.txt")))
        var line = reader.readLine()
        while(line != null) {
            labelList.add(line)
            line = reader.readLine()
        }
        outputs = Array(1) { FloatArray(labelList.size) }
    }

    private fun getNumLabels(): Int {
        return labelList.size
    }

    private fun getBitmap(imageName: String): Bitmap {
        val stream = BitmapFactory.decodeStream(assets.open(imageName))
        runOnUiThread {
            binding.imageView.setImageBitmap(Bitmap.createScaledBitmap(stream, 480, 480, true))
        }
        return Bitmap.createScaledBitmap(stream, 224, 224, true)
    }

    fun onNNApiClick(v: View) {
    }
}
