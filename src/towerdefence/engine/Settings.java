package towerdefence.engine;

import org.newdawn.slick.SlickException;
import org.newdawn.slick.util.xml.SlickXMLException;
import org.newdawn.slick.util.xml.XMLElement;
import org.newdawn.slick.util.xml.XMLElementList;
import org.newdawn.slick.util.xml.XMLParser;

/**
 * Load game setttings from settings.xml
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class Settings {
    
    private final XMLElement root;
    
    // Player settings
    private int startingMoney;
    private int playerHealth;
    
    // Critter settings
    private int[] critterHealth = new int[3];
    private double[] critterSpeed = new double[3];
    private int[] reward = new int[3];
    
    // Tower settings
    private int[] baseDPS = new int[3];
    private int[] range = new int[3];
    private boolean[] lockOn = new boolean[3];
    private int[] cost = new int[3];
    
    public Settings() throws SlickException {
        XMLParser parser = new XMLParser();

        root = parser.parse("data/settings.xml");
        
        loadPlayerSettings();
        loadCritterSettings();
        loadTowerSettings();
        
    }
    
    private void loadPlayerSettings() throws SlickXMLException {
        XMLElement player = root.getChildrenByName("Player").get(0);
        startingMoney = player.getIntAttribute("startingMoney");
        playerHealth = player.getIntAttribute("health");
    }

    private void loadCritterSettings() throws SlickXMLException {
        
        XMLElementList critters = root.getChildrenByName("Critter").get(0).getChildren();
        for(int i=0;i<critters.size();i++) {
            critterHealth[i]=critters.get(i).getIntAttribute("health");
            critterSpeed[i]=critters.get(i).getDoubleAttribute("speed");
            reward[i]=critters.get(i).getIntAttribute("reward");
        }
    }
    
    private void loadTowerSettings() throws SlickXMLException {
        
        XMLElementList towers = root.getChildrenByName("Tower").get(0).getChildren();
        for(int i=0;i<towers.size();i++) {
            baseDPS[i]=towers.get(i).getIntAttribute("baseDPS");
            range[i]=towers.get(i).getIntAttribute("range");
            lockOn[i]=towers.get(i).getBooleanAttribute("lockOn");
            cost[i]=towers.get(i).getIntAttribute("cost");
        }
    }

    /**
     * @return the startingMoney
     */
    public int getStartingMoney() {
        return startingMoney;
    }

    /**
     * @return the playerHealth
     */
    public int getPlayerHealth() {
        return playerHealth;
    }

    /**
     * @return the critterHealth
     */
    public int[] getCritterHealth() {
        return critterHealth;
    }

    /**
     * @return the critterSpeed
     */
    public double[] getCritterSpeed() {
        return critterSpeed;
    }

    /**
     * @return the baseDPS
     */
    public int[] getBaseDPS() {
        return baseDPS;
    }

    /**
     * @return the range
     */
    public int[] getRange() {
        return range;
    }

    /**
     * @return the lockOn
     */
    public boolean[] getLockOn() {
        return lockOn;
    }

    /**
     * @return the reward
     */
    public int[] getReward() {
        return reward;
    }

    /**
     * @return the cost
     */
    public int[] getCost() {
        return cost;
    }


    
    
}
