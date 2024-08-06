package com.mobileml.shubham0204.scikitlearndemo

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.util.Log

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize the views
        val outputTextView = findViewById<TextView>(R.id.output_textview)
        val button = findViewById<Button>(R.id.predict_button)

        button.setOnClickListener {
            // Parse input from inputEditText
            val startTimeCreate = System.currentTimeMillis()
            val ortEnvironment = OrtEnvironment.getEnvironment()
            val ortSession = createORTSession(ortEnvironment)
            val endTimeCreate = System.currentTimeMillis()
            Log.d("ONNX", "Time to create ORT session: ${endTimeCreate - startTimeCreate} ms")

            val startTimePredict = System.currentTimeMillis()
            val output = runPrediction(ortSession, ortEnvironment)
            val endTimePredict = System.currentTimeMillis()
            outputTextView.text =
                "Time to load the model: ${endTimeCreate - startTimeCreate} ms, \n Time to run prediction: ${endTimePredict - startTimePredict} ms"

        }

    }

    // Create an OrtSession
    private fun createORTSession(ortEnvironment: OrtEnvironment): OrtSession {
        val modelBytes = resources.openRawResource(R.raw.tinybert_model).readBytes()
        return ortEnvironment.createSession(modelBytes)
    }


    // Make predictions with given inputs
    private fun runPrediction(ortSession: OrtSession, ortEnvironment: OrtEnvironment) {
        // Get the names of the input nodes
        val inputNames = ortSession.inputNames
        val inputName = inputNames.find { it.contains("input_ids") }
            ?: throw IllegalArgumentException("input_ids not found")
        val attentionName = inputNames.find { it.contains("attention_mask") }
            ?: throw IllegalArgumentException("attention_mask not found")

        // Adjust token IDs and attention mask to the required sequence length
        val sequenceLength = 6
        val inputIds = longArrayOf(101L, 7592L, 999L, 1L, 1L, 1L)
        val attentionMask = longArrayOf(1L, 1L, 1L, 0L, 0L, 0L) // Adjust as needed

        // Create input tensors
        val inputIdsTensor = OnnxTensor.createTensor(ortEnvironment, arrayOf(inputIds))
        val attentionMaskTensor = OnnxTensor.createTensor(ortEnvironment, arrayOf(attentionMask))

        // Run the model
        val results =
            ortSession.run(mapOf(inputName to inputIdsTensor, attentionName to attentionMaskTensor))

        return
    }


}
