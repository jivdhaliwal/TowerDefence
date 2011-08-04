package towerdefence.engine;

import java.util.ArrayList;

import org.newdawn.slick.SlickException;
import org.newdawn.slick.util.xml.SlickXMLException;
import org.newdawn.slick.util.xml.XMLElement;
import org.newdawn.slick.util.xml.XMLElementList;
import org.newdawn.slick.util.xml.XMLParser;

import towerdefence.engine.entity.Critter;
import towerdefence.engine.entity.Tower;

/**
 * Load game setttings from settings.xml
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class Settings {
	
	private static Settings settings = null;
    
    private final XMLElement root;
    
    private static final int numTowers=5;
    private static final int numCritters=3;
    
    // Player settings
    private int startingMoney;
    private int playerHealth;
    
    // Critter settings
    private float[] minCritterHealth = new float[numCritters];
    private float[] critterHealth = new float[numCritters];
    private double[] critterSpeed = new double[numCritters];
    private int[] reward = new int[numCritters];
    
    // Tower settings
    private int[] baseDPS = new int[numTowers];
    private int[] range = new int[numTowers];
    private int[] shootingCounter = new int[numTowers];
    private boolean[] lockOn = new boolean[numTowers];
    private int[] cost = new int[numTowers];
    
    public Settings() throws SlickException {
        XMLParser parser = new XMLParser();

        root = parser.parse("settings.xml");
        
        loadPlayerSettings();
        loadCritterSettings();
        loadTowerSettings();
        
    }
    
    public static Settings getInstance() {
        if(settings==null) { 
            try {
				settings = new Settings();
			} catch (SlickException e) {
				e.printStackTrace();
			}
        }
        return settings;
    }
    
    private void loadPlayerSettings() throws SlickXMLException {
        XMLElement player = root.getChildrenByName("Player").get(0);
        startingMoney = player.getIntAttribute("startingMoney");
        playerHealth = player.getIntAttribute("health");
    }

    private void loadCritterSettings() throws SlickXMLException {
        
        XMLElementList critters = root.getChildrenByName("Critter").get(0).getChildren();
        for(int i=0;i<critters.size();i++) {
            critterHealth[i]=(float)critters.get(i).getIntAttribute("health");
            minCritterHealth[i]=(float)critters.get(i).getIntAttribute("health");
            critterSpeed[i]=critters.get(i).getDoubleAttribute("speed");
            reward[i]=critters.get(i).getIntAttribute("reward");
        }
    }
    
    private void loadTowerSettings() throws SlickXMLException {
        
        XMLElementList towers = root.getChildrenByName("Tower").get(0).getChildren();
        for(int i=0;i<towers.size();i++) {
            baseDPS[i]=towers.get(i).getIntAttribute("baseDPS");
            range[i]=towers.get(i).getIntAttribute("range");
            shootingCounter[i]=towers.get(i).getIntAttribute("shootingCounter");
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
    public float[] getCritterHealth() {
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

    public int[] getShootingCounter() {
		return shootingCounter;
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

    
    /*
     * Based on the number of active towers in the game
     * scale the health of critters
     */
	public void updateDamageScaling(ArrayList<Tower> towerList) {
		float healthModifier = 0f;
		for (Tower tower : towerList) {
			switch (tower.getType()) {
				case Tower.NORMAL:
					healthModifier += 6f;
					break;
				case Tower.FIRE:
					healthModifier += 7f;
					break;
				case Tower.ICE:
					healthModifier += 8f;
					break;
				case Tower.BULLET:
					healthModifier += 3.5f;
					break;
				case Tower.ROCKET:
					healthModifier += 8f;
					break;
			}
		}
		for (int i = 0; i < numCritters; i++) {
			critterHealth[i] = minCritterHealth[i] + (healthModifier);
		}
	}

    
    
}
