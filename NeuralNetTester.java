import java.util.*;
public class NeuralNetTester {

   public static void main(String[] args) {
      
      ArrayList<Integer> layerSizes = new ArrayList<>();
      layerSizes.add(2);
      layerSizes.add(2);
      NeuralNet newNetwork = new NeuralNet(layerSizes, 0.000001);
   
   }

}