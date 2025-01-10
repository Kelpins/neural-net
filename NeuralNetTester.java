import java.util.*;
public class NeuralNetTester {

   public static void main(String[] args) {
      
      ArrayList<Integer> layerSizes = new ArrayList<>();
      layerSizes.add(2);
      layerSizes.add(2);
      NeuralNet newNetwork = new NeuralNet(layerSizes, 0.000001);

      Map<double[], double[]> trainingData = new HashMap<>;
      for (int i = 1; i < 100; i++) {
         double[] inputs = {i * Math.random(), i * Math.random()};
         double[] outputs = {inputs[0]+inputs[1], inputs[0]-inputs[1], inputs[0]*inputs[1], inputs[0]/inputs[1]};
         trainingData.put(inputs, outputs);
      }
      
      Map<double[], double[]> testingData = new HashMap<>;
      for (int i = -1000; i < -1; i++) {
         double[] inputs = {i * Math.random(), i * Math.random()};
         double[] outputs = {inputs[0]+inputs[1], inputs[0]-inputs[1], inputs[0]*inputs[1], inputs[0]/inputs[1]};
         trainingData.put(inputs, outputs);
      }

   }

}
