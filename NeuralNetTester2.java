import java.util.*;
public class NeuralNetTester2 {

   public static void main(String[] args) {
   
      HashMap<ArrayList<Double>, Double> train = new HashMap<>();
      HashMap<ArrayList<Double>, Double> test = new HashMap<>();
      
      int trainSize = 5;
      int testSize = 2;
      
      for (int i = 0; i < trainSize; i++) {
         
         ArrayList mapInputs = new ArrayList<>();
         double num1 = Math.round(Math.random() * 1000);
         double num2 = Math.round(Math.random() * 1000);
         mapInputs.add(num1);
         mapInputs.add(num2);
         
         train.put(mapInputs, num1+num2);
      
      }
      
      for (int i = 0; i < testSize; i++) {
         
         ArrayList mapInputs = new ArrayList<>();
         double num1 = Math.round(Math.random() * 1000);
         double num2 = Math.round(Math.random() * 1000);
         mapInputs.add(num1);
         mapInputs.add(num2);
         
         test.put(mapInputs, num1+num2);
      
      }
            
      ArrayList<Integer> layerSizes = new ArrayList<>();
      layerSizes.add(3);
      layerSizes.add(4);
      layerSizes.add(4);
      layerSizes.add(4);
      layerSizes.add(2);
      
      NeuralNet2 newNetwork = new NeuralNet2(layerSizes, 0.00000000015);
      
      System.out.println(newNetwork.test(test));
      
      newNetwork.train(train, 15);
      
      System.out.println(newNetwork.test(test));

   }

}
