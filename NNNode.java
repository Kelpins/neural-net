import java.util.*;

/**
 * A node in a neural network, holding a bias and a list of incoming weights.
 * @author Amanda O'Neal
 */
public class NNNode {
    private double bias;
    private List<Double> weights;
    private double numWeights;
    private double error;
    private double activation;

    /**
     * Constructs a NNNode with random bias and weights.
     *
     * @param numWeights The number of inbound weights for this node.
     */
    public NNNode(int numWeights) {
        this.bias = Math.random() * 0.05;
        this.weights = new ArrayList<>();
        for (int i = 0; i < numWeights; i++) {
            this.weights.add(Math.random() * 0.05);
        }
        this.error = 0.0;
        this.activation = 0.0;
        this.numWeights = numWeights;
    }

    /**
     * Gets the bias value for this node.
     *
     * @return The bias value of the node.
     */
    public double getBias() {
        return bias;
    }

    /**
     * Sets a new bias value for this node.
     *
     * @param bias The new bias value to be set.
     */
    public void setBias(double bias) {
        this.bias = bias;
    }

    /**
     * Gets the weight at a specific index.
     *
     * @param index The index of the weight to retrieve.
     * @return The weight at the specified index
     */
    public double getWeight(int index) {
        return this.weights.get(index);
    }
    
    /**
     * Sets the weight at a specific index.
     *
     * @param index The index of the weight to be updated.
     * @param weight The new weight value to set.
     */
    public void setWeight(int index, double weight) {
        this.weights.set(index, weight);
    }
    
    public double getError() {
    
      return this.error;
    
    }
    
    public void setError(double e) {
    
      this.error = e;
    
    }
    
    public void setActivation(double a) {
    
      this.activation = a;
    
    }
    
    public double getActivation() {
    
      return this.activation;
    
    }
    
    public double getWeightSum() {
    
      double sum = 0.0;
    
      for (double weight : this.weights) {
      
         sum += weight;
      
      }
      
      return sum;
      
    }
    
    public double getNumWeights() {
    
      return this.numWeights;
    
    }
    
}
