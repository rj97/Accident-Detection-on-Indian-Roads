package ai.movehack.roadaccidents;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

/**
 * A classifier specialized to label images using TensorFlow.
 */
public class TensorFlowImageClassifier implements Classifier {

    // Only return this many results with at least this confidence.
    private static final int MAX_RESULTS = 3;
    private static final float THRESHOLD = 0.6f; //Return results >= 60%

    // Config values.
    private String inputName;
    private String outputName;
    private int inputSize;
    private int imageMean;
    private float imageStd;

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;
    private byte[] bytesValues;
    private float[] outputs;
    private float[] output_classes;
    private String[] outputNames;

    private TensorFlowInferenceInterface inferenceInterface;

    private boolean runStats = false;

    private TensorFlowImageClassifier() {
    }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     * @param inputSize     The input size. A square image of inputSize x inputSize is assumed.
     * @param inputName     The label of the image input node.
     * @param outputName    The label of the output node.
     * @throws IOException
     */
    public static Classifier create(
            AssetManager assetManager,
            String modelFilename,
            String labelFilename,
            int inputSize,
            String inputName,
            String outputName)
            throws IOException {
        TensorFlowImageClassifier c = new TensorFlowImageClassifier();
        c.inputName = inputName;
        c.outputName = outputName;

        // Read the label names into memory.
        // TODO(andrewharp): make this handle non-assets.
        String actualFilename = labelFilename.split("file:///android_asset/")[1];

        BufferedReader br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));

        String line;
        while ((line = br.readLine()) != null) {
            c.labels.add(line);
        }

        br.close();

        c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);
        // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
        //int numClasses = (int) c.inferenceInterface.graph().operation(outputName).output(0).shape().size(1);
        //int numClasses = (int) c.inferenceInterface.graph().operation("detection_classes").output(0).shape().size(1);
        int numClasses = 1;

        // Ideally, inputSize could have been retrieved from the shape of the input operation.  Alas,
        // the placeholder node for input in the graphdef typically used does not specify a shape, so it
        // must be passed in as a parameter.
        c.inputSize = inputSize;

        // Pre-allocate buffers.
        //c.outputNames = new String[]{"detection_boxes","detection_scores","detection_classes","num_detections"};
        c.outputNames = new String[]{"detection_scores","detection_classes"};
        c.intValues = new int[inputSize * inputSize];
        c.bytesValues = new byte[inputSize * inputSize * 3];
        //c.outputs = new float[numClasses];
        c.outputs = new float[20];
        c.output_classes = new float[20];

        return c;
    }

    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            bytesValues[i * 3] = (byte) ((val >> 16) & 0xFF);
            bytesValues[i * 3 + 1] = (byte) ((val >> 8) & 0xFF);
            bytesValues[i * 3 + 2] = (byte) (val & 0xFF);
        }

        // Copy the input data into TensorFlow.
        inferenceInterface.feed(inputName, bytesValues, new long[]{1, inputSize, inputSize, 3});

        // Run the inference call.
        inferenceInterface.run(outputNames, runStats);

        // Copy the output Tensor back into the output array.
        inferenceInterface.fetch("detection_scores", outputs);
        inferenceInterface.fetch("detection_classes", output_classes);

        // Find the best classifications.
        /*PriorityQueue<Recognition> pq = new PriorityQueue<Recognition>(
                        3,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });*/
        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        for (int i = 0; i < outputs.length; i++) {
            if (outputs[i] > THRESHOLD) {
                //pq.add(new Recognition("" + i, labels.size() > i ? labels.get(i) : "unknown", outputs[i], null));
                recognitions.add(new Recognition("" + i, labels.get(Math.round(output_classes[i])), outputs[i], null));
            }
        }
        /*int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }*/
        return recognitions;
    }

    @Override
    public void enableStatLogging(boolean debug) {
        runStats = debug;
    }

    @Override
    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }
}
