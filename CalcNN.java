import java.util.*;

/**
   A neural network which is able to add one to a number. 
   Sample input: 1; sample output: 2.
*/
class CalcNN {
   /* Declare instance variables: 
      an NNLayer for your hidden layer; 
      an NNNode for your output node; and
      a double for your learning rate */
   private NNLayer[] hiddenLayers;
   private NNNode outputNode;
   private double learningRate;
   private int numNodesPerLayer;
   
    public CalcNN(int numHiddenLayers, int numHiddenNodes, double learningRate) {
      /* Initialize the three instance variables.
         The hidden layer should have one incoming weight per node.
         The output node should have numHiddenNodes incoming weights per node.
        */
      this.hiddenLayers = new NNLayer[numHiddenLayers];
      for (int i = 0; i < numHiddenLayers; i++) {
         this.hiddenLayers[i] = new NNLayer(numHiddenNodes, 1);
      }
      this.outputNode = new NNNode(numHiddenNodes);
      this.learningRate = learningRate;
      this.numNodesPerLayer = numHiddenNodes;
      
    }

    public double test(Map<Double, Double> testData) {
      /*
         Goal: run each of our test inputs through the network. calculate accuracy of outputs via root mean squared error.
         
         start an error variable at zero. this will calculate the sum of squared errors.
         for each key/value pair in the test data map:
            "feed forward": run the input through the neural network to get the output.
                  reminder: hidden node activation = activationFunction(bias + weight * input)
                      output = bias + weight * activation for hidden node #1 + weight * activation for hidden node #2 + ...
            calculate the squared error: (predicted output - correct output)^2. add this to your error variable.
            optional: for debugging/learning purposes, print your input, expected output, and predicted output
         
         return the error: square root of (your error sum divided by the number of inputs in your test data set)         
      */
      double errorSum = 0;
      
      for (Double key : testData.keySet()) { //for each test val
         //feed fwd
         double input = key;
         double output = this.outputNode.getBias();
         
         double[][] allActivations = new double[this.hiddenLayers.length+1][numNodesPerLayer];
         for (int k = 0; k < allActivations[0].length; k++) {
            allActivations[0][k] = input; //first layer input layer
         }
         
         for (int i = 1; i < this.hiddenLayers.length+1; i++) {
            NNLayer currentHiddenLayer = this.hiddenLayers[i-1];
            
            for (int j = 0; j < numNodesPerLayer; j++) { //each hidden node's activation val
               NNNode node = currentHiddenLayer.getNodes().get(j);
               allActivations[i][j] = activationFunction(node.getBias() + node.getWeight(0)*allActivations[i-1][j]);
            }
         }
         //output = sum of w(each) * allActivations[last][each];
            for (int a = 0; a < allActivations[0].length; a++) {
               output += this.outputNode.getWeight(a) * allActivations[this.hiddenLayers.length - 1][a];
            }
         errorSum += Math.pow((testData.get(key) - output), 2); //running sum of squared error
      }
      return Math.pow(errorSum/testData.size(), 0.5); //return sqrt of average squared error -> avg error
    }
    private double activationFunction(double value) {
      return value;
    }

    public void train(Map<Double, Double> trainingData, int epochs) {
       /* DO THIS METHOD LAST! Check in with your teacher before starting this method. */
       
       /*
            for each epoch:
               for each key/value pair in the training data map:
                  "feed forward": run the input through the neural network to get the output. HOWEVER,
                     this version, unlike your test() version, should save the hidden activations information -
                     we will need them later. consider storing them in a double[] hiddenActivations.
                     
                   "backpropogate": go backwards through the network (starting at output) and adjust weights/biases:
                     (note: I'm recommending you do multiple loops for this. It could technically be done with one loop,
                              but doing it with multiple is much more extensible for if you want to add more hidden layers later.)
         
                     first, update the output node's weights and bias:
                     calculate the output error (expected output - actual output from the feed forward) 
                     update the output node's bias: bias = bias + output error * learning rate
                     for each hidden layer node:
                         set the weight between that node and the output to: its previous weight + output error * learning rate * activation for that hidden node
                         

                     then, update the hidden layer node's weights and biases:
                     for each hidden layer node:
                         calculate the node's error: output error * weight from hidden node to output.
                         update the weight from input to this node to: previous weight + learning rate * node's error * input
                         update the bias for this node to: previous bias + learning rate * node's error
       */      
      
      for (Double key : trainingData.keySet()) { //for each test val
         //feed fwd
         double input = key;
         double output = this.outputNode.getBias();
         
         double[][] allActivations = new double[this.hiddenLayers.length+1][numNodesPerLayer];
         for (int k = 0; k < allActivations[0].length; k++) {
            allActivations[0][k] = input; //first layer input layer
         }
         
         for (int i = 1; i < this.hiddenLayers.length+1; i++) {
            NNLayer currentHiddenLayer = this.hiddenLayers[i-1];
            
            for (int j = 0; j < numNodesPerLayer; j++) { //each hidden node's activation val
               NNNode node = currentHiddenLayer.getNodes().get(j);
               allActivations[i][j] = activationFunction(node.getBias() + node.getWeight(0)*allActivations[i-1][j]);
            }
         }
         //output = sum of w(each) * allActivations[last][each];
            for (int a = 0; a < allActivations[0].length; a++) {
               output += this.outputNode.getWeight(a) * allActivations[this.hiddenLayers.length - 1][a];
            }
         
         //backprop: adjusting output node values
         double outputError = trainingData.get(key) - output;
         this.outputNode.setBias(this.outputNode.getBias() + outputError * this.learningRate); //fix output bias
         
         for (int i = 0; i < activations.size(); i++) {
         
            NNLayer currentLayer = hiddenLayers[i];
            
            for (int j = 0; j < currentLayer.getNodes().size(); j++) {
            
               NNNode currentNode = currentLayer.getNodes().get(j);
               
               this.outputNode.setWeight(j, currentNode.getWeight(0) + outputError * this.learningRate 
                  * activations.get(i)); //fix weight btwn output and hidden
                  
               double currentError = outputError * outputNode.getWeight(i); //hidden node error val
               currentNode.setWeight(0, currentNode.getWeight(0) + this.learningRate * currentError * input); 
                  //fix weight btwn input and hidden
               currentNode.setBias(currentNode.getBias() + this.learningRate * currentError); //fix hidden bias
            }
         }

      }
    }
