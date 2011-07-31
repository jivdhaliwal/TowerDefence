package towerdefence.engine;

/**
 *
 * Handles in-game currency
 * 
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class Player {
    
    private static Player player = null;
    
    private int cash;
    private int health;
    
    private int[] critterReward;
    private int[] towerCost;

    private Player() {
    	cash = Settings.getInstance().getStartingMoney();
    	health = Settings.getInstance().getPlayerHealth();
    	critterReward = Settings.getInstance().getReward();
    	towerCost = Settings.getInstance().getCost();
    }
    
    public static Player getInstance() {
        if(player==null) { 
            player = new Player();
        }
        return player;
    }
    
    public void resetParams() {
    	player = new Player();
    }

    /**
     * @return the cash
     */
    public int getCash() {
        return cash;
    }

    /**
     * @param cash the cash to set
     */
    public void setCash(int cash) {
        this.cash = cash;
    }
    
    /**
     * 
     * Add cash to the wallet based on which critter type killed
     * 
     * @param type Critter type
     */
    public void killCritter(int type) {
        setCash(getCash() + critterReward[type]);
    }
    
    public void addTower(int type) {
        setCash(getCash() - getTowerCost()[type]);
    }
    
    public void sellTower(int type) {
        setCash(getCash() + (getTowerCost()[type])/2);
    }

    /**
     * @param critterReward the critterReward to set
     */
    public void setCritterReward(int[] critterReward) {
        this.critterReward = critterReward;
    }

    /**
     * @param towerCost the towerCost to set
     */
    public void setTowerCost(int[] towerCost) {
        this.towerCost = towerCost;
    }

    /**
     * @return the towerCost
     */
    public int getTowerCost(int type) {
        return getTowerCost()[type];
    }

    /**
     * @return the health
     */
    public int getHealth() {
        return health;
    }

    /**
     * @param health the health to set
     */
    public void setHealth(int health) {
        this.health = health;
    }

    /**
     * @return the towerCost
     */
    public int[] getTowerCost() {
        return towerCost;
    }

    
}
