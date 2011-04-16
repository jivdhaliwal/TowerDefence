package towerdefence.engine.levelLoader;

/**
 *
 * 
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class Wave {
    private final int critterType;
    private final int numCritters;
    private final int timeToWait;

    public Wave(int critterType, int numCritters, int timeToWait) {
        this.critterType = critterType;
        this.numCritters = numCritters;
        this.timeToWait = timeToWait;
    }

    public int getCritterType() {
        return critterType;
    }

    public int getNumCritters() {
        return numCritters;
    }

    public int timeToWait() {
        return timeToWait;
    }

}
