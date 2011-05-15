package towerdefence.engine;

import towerdefence.GameplayState;

/**
 *
 * Handles in-game currency
 * 
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class Wallet {
    
    private static Wallet wallet = null;
    
    private int cash;
    
    private int[] critterReward;
    private int[] towerCost;

    private Wallet() {
        cash = 0;
    }
    
    public static Wallet getInstance() {
        if(wallet==null) { 
            wallet = new Wallet();
        }
        return wallet;
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
        cash+=critterReward[type];
    }
    
    public void addTower(int type) {
        cash-=towerCost[type];
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
        return towerCost[type];
    }
    
}
