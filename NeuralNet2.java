import java.util.*;

public class NeuralNet2 {
   
   private ArrayList<NNLayer> hiddenLayers;
   private NNNode outNode;
   private double learningRate;

   public NeuralNet2(ArrayList<Integer> layerSizes, double learningRate) {
      
      this.hiddenLayers = new ArrayList<NNLayer>();
      
      for (int i = 1; i < layerSizes.size(); i++) {
      
         this.hiddenLayers.add(new NNLayer(layerSizes.get(i), layerSizes.get(i-1)));
      
      }
      
      this.outNode = new NNNode(layerSizes.get(layerSizes.size()-1));
      this.learningRate = learningRate;
      
   }
   
   public double test(Map<ArrayList<Double>, Double> testData) {
   
      double error = 0.0;
      
      for (ArrayList<Double> key : testData.keySet()) {
      
         error += Math.pow(feedForward(key, testData.get(key)), 2);
      
      }
      
      return Math.sqrt(error / testData.size());
          
   }
   
   public void train(HashMap<ArrayList<Double>, Double> trainData, int epochs) {
   
      for (int e = 0; e < epochs; e++) {
         
         //System.out.println("epoch " + e);
      
         for (ArrayList<Double> key : trainData.keySet()) {
            
            double error = feedForward(key, trainData.get(key));
            backPropogate(key, error);
         
         }
         
      }
   
   }
   
   private double feedForward(ArrayList<Double> inputValues, double expectedOut) {
      
      NNLayer firstLayer = this.hiddenLayers.get(0);
      
      for (int i = 0; i < firstLayer.getNodes().size(); i++) {
         
         NNNode currentNode = firstLayer.getNodes().get(i);
                     
         for (int j = 0; j < 2; j++) {
         
            currentNode.setActivation(currentNode.getActivation() + inputValues.get(j) * currentNode.getWeight(j));
         
         }
               
      }
      
      for (int i = 1; i < this.hiddenLayers.size(); i++) {
      
         for (int j = 0; j < this.hiddenLayers.get(i).getNodes().size(); j++) {
         
            NNNode currentNode = this.hiddenLayers.get(i).getNodes().get(j);
            
            for (int k = 0; k < currentNode.getNumWeights(); k++) {
               
               NNLayer previousLayer = this.hiddenLayers.get(i-1);
               
               currentNode.setActivation(currentNode.getActivation() + previousLayer.getNodes().get(k).getActivation() * currentNode.getWeight(k));
            
            }
            
            currentNode.setActivation(currentNode.getActivation() + currentNode.getBias());
         
         }
      
      }
      
      double actualOut = 0.0;
      NNLayer finalLayer = this.hiddenLayers.get(this.hiddenLayers.size()-1);
      
      for (int i = 0; i < finalLayer.getNodes().size(); i++) {
          
         NNNode currentNode = finalLayer.getNodes().get(i);
         double activation = currentNode.getActivation();
         double weight = this.outNode.getWeight(i);
         actualOut += (activation * weight);
      
      }
      
      actualOut += this.outNode.getBias();
      
      //System.out.println("actual: " + actualOut + ", expected: " + expectedOut + ", error: " + (expectedOut-actualOut));

      return expectedOut - actualOut;
   
   }
   
   private void backPropogate(ArrayList<Double> inputValues, double error) {
      
      this.outNode.setBias(this.outNode.getBias() + this.learningRate * error);
      
      List<NNNode> finalLayer = this.hiddenLayers.get(this.hiddenLayers.size()-1).getNodes();
      
      for (int a = 0; a < finalLayer.size(); a++) {
      
         NNNode currentNode = finalLayer.get(a);
         this.outNode.setWeight(a, this.outNode.getWeight(a) + this.learningRate * error * currentNode.getActivation());
         currentNode.setError(this.outNode.getWeight(a) * error);
         
      }
      
      for (int i = this.hiddenLayers.size()-1; i > -1; i--) {
         
         NNLayer currentLayer = this.hiddenLayers.get(i);
         NNLayer futureLayer = new NNLayer(0, 0);
         
         try {
         
            futureLayer = this.hiddenLayers.get(i+1);
         
         }
         catch (Exception e) {
            
            List<NNNode> relevantNodes = new ArrayList<NNNode>();
            relevantNodes.add(this.outNode);
         
            futureLayer = new NNLayer(relevantNodes);
                     
         }
      
         for (int j = 0; j < currentLayer.getNodes().size(); j++) {
         
            NNNode currentNode = currentLayer.getNodes().get(j);
            currentNode.setError(currentNode.getError() / futureLayer.getNodes().size());
            currentNode.setBias(currentNode.getBias() + this.learningRate * currentNode.getError());
            
            //System.out.println(currentNode.getError());
            
            try {
            
               NNLayer previousLayer = this.hiddenLayers.get(i-1);
                        
               for (int k = 0; k < previousLayer.getNodes().size(); k++) {
               
                  NNNode previousNode = previousLayer.getNodes().get(k);
                  
                  currentNode.setWeight(k, currentNode.getWeight(k) + this.learningRate * currentNode.getError() * previousNode.getActivation());
                  
                  previousNode.setError(previousNode.getError() + (currentNode.getError() * currentNode.getWeight(k)));// * currentNode.getWeight(k) / currentNode.getWeightSum()));
               
               }
            
            }
            catch (Exception e) {
            
               for (int m = 0; m < inputValues.size(); m++) {
               
                  currentNode.setWeight(m, currentNode.getWeight(m) + this.learningRate * currentNode.getError() * inputValues.get(m));
               
               }
            
            }
                  
         }
      
      }
         
   }

}