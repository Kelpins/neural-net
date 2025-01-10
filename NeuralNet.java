import java.util.*;

public class NeuralNet {

   //for each hidden layer
      //for each node in hidden layer
         //for i in range(len(previous connection))
            // += weight * previous node value
         //add bias
   
   
   private ArrayList<NNLayer> hiddenLayers;
   private ArrayList<Integer> hiddenLayerSizes;
   private NNNode outNode;
   private double learningRate;

   public NeuralNet(ArrayList<Integer> layerSizes, double learningRate) {
      
      for (int i = 1; i < layerSizes.size(); i++) {
      
         this.hiddenLayers.add(new NNLayer(layerSizes.get(i), layerSizes.get(i-1)));
      
      } 
      this.outNode = new NNNode(layerSizes.get(layerSizes.size()-1));
      this.learningRate = learningRate;
      
   }

}