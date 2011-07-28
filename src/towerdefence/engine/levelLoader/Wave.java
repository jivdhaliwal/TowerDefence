package towerdefence.engine.levelLoader;

/**
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class Wave {
    private final int critterType;
    private final int numCritters;
    private final int timeToSpawn;
    private final int timeToWait;

    public Wave(int critterType, int numCritters, int timeToSpawn, int timeToWait) {
        this.critterType = critterType;
        this.numCritters = numCritters;
        this.timeToSpawn = timeToSpawn;
        this.timeToWait = timeToWait;
    }

    public int getTimeToSpawn() {
		return timeToSpawn;
	}

	/**
     * @return the critterType
     */
    public int getCritterType() {
        return critterType;
    }

    /**
     * @return the numCritters
     */
    public int getNumCritters() {
        return numCritters;
    }

    /**
     * @return the timeToWait
     */
    public int getTimeToWait() {
        return timeToWait;
    }



}
