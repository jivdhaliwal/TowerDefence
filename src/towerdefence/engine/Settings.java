/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package towerdefence.engine;

import org.newdawn.slick.SlickException;
import org.newdawn.slick.util.xml.SlickXMLException;
import org.newdawn.slick.util.xml.XMLElement;
import org.newdawn.slick.util.xml.XMLElementList;
import org.newdawn.slick.util.xml.XMLParser;

/**
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class Settings {
    
    private final XMLElement root;
    
    private int startingMoney;
    private int playerHealth;
    private int[] critterHealth = new int[3];
    private double[] critterSpeed = new double[3];
    private int[] baseDPS = new int[3];
    private int[] range = new int[3];
    private boolean[] lockOn = new boolean[3];
    
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
        }
    }
    
    private void loadTowerSettings() throws SlickXMLException {
        
        XMLElementList towers = root.getChildrenByName("Tower").get(0).getChildren();
        for(int i=0;i<towers.size();i++) {
            baseDPS[i]=towers.get(i).getIntAttribute("baseDPS");
            range[i]=towers.get(i).getIntAttribute("range");
            lockOn[i]=towers.get(i).getBooleanAttribute("lockOn");
        }
    }

    /**
     * @return the critterHealth
     */
    public int getCritterHealth(int type) {
        return critterHealth[type];
    }

    /**
     * @return the critterSpeed
     */
    public double getCritterSpeed(int type) {
        return critterSpeed[type];
    }

    /**
     * @return the baseDPS
     */
    public int getBaseDPS(int type) {
        return baseDPS[type];
    }

    /**
     * @return the range
     */
    public int getRange(int type) {
        return range[type];
    }

    /**
     * @return the lockOn
     */
    public boolean getLockOn(int type) {
        return lockOn[type];
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
    
    
}
