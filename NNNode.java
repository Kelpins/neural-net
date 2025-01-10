import java.util.*;

/**
 * A node in a neural network, holding a bias and a list of incoming weights.
 * @author Amanda O'Neal
 */
public class NNNode {
    private double bias;
    private List<Double> weights;

    /**
     * Constructs a NNNode with random bias and weights.
     *
     * @param numWeights The number of inbound weights for this node.
     */
    public NNNode(int numWeights) {
        this.bias = Math.random() * 0.1;
        this.weights = new ArrayList<>();
        for (int i = 0; i < numWeights; i++) {
            this.weights.add(Math.random() * 0.1);
        }
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
}
