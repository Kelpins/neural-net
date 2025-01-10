import java.util.*;

/**
 * A layer in a neural network consisting of multiple NNNodes.
 * @author Amanda O'Neal
 */
public class NNLayer {
    private List<NNNode> nodes;

    /**
     * Constructs a NNLayer with a specified number of nodes, each initialized with a given number of incoming weights.
     *
     * @param numNodes The number of nodes in the layer.
     * @param numWeightsPerNode The number of incoming weights each node should have.
     */
    public NNLayer(int numNodes, int numWeightsPerNode) {
        this.nodes = new ArrayList<>();
        for (int i = 0; i < numNodes; i++) {
            this.nodes.add(new NNNode(numWeightsPerNode));
        }
    }

    /**
     * Gets the list of nodes in this layer.
     *
     * @return A copy of the list of nodes.
     */
    public List<NNNode> getNodes() {
        return nodes;
    }
}
