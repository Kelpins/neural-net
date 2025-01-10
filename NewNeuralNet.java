public class NewNeuralNet {

   //for each hidden layer
      //for each node in hidden layer
         //for i in range(len(previous connection))
            // += weight * previous node value
         //add bias

   private NNLayer hiddenLayer;
   private NNNode outNode;
   private double learningRate;

   public AddOneNN(int numHiddenNodes, double learningRate) {
   
      this.hiddenLayer = new NNLayer(numHiddenNodes, 1);
      this.outNode = new NNNode(numHiddenNodes);
      this.learningRate = learningRate;
      
   }

}
[[100, 0, 75, 50], [], []] -> 65