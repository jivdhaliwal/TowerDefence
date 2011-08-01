/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package towerdefence;

import java.util.ArrayList;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.state.StateBasedGame;
import towerdefence.engine.Player;
import towerdefence.engine.ResourceManager;
import towerdefence.engine.component.ImageRenderComponent;
import towerdefence.engine.component.LaserTowerComponent;
import towerdefence.engine.entity.*;

/**
 *
 * Handles the list of towers
 * Includes adding to, deleting from the list
 * Stores an updated list of the critters in the map
 * Handles the update and render methods for towers
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class TowerManager {
    
    private ArrayList<Tower> towerList = new ArrayList<Tower>();
    private ArrayList<Critter> critterList;

    /*
     * Default Tower when no type is defined
     */
    public void addTower(String id, Vector2f position, int type, boolean placed, boolean active) throws SlickException {
    	
        // Check if player can afford to add the tower
        if(Player.getInstance().getCash()-Player.getInstance().getTowerCost(type) >=0) {
            Player.getInstance().addTower(type);
            Tower tower = new Tower(id, true);
            tower.setPosition(position);
            tower.setType(type);
            tower.setIsPlaced(placed);
            tower.setActive(active);
            switch (type) {
    		case Tower.NORMAL:
    			tower.AddComponent(new ImageRenderComponent("TowerRender",
    					ResourceManager.getInstance().getImage("NORMAL_TOWER")));
    			break;
    		case Tower.FIRE:
    			tower.AddComponent(new ImageRenderComponent("TowerRender",
    					ResourceManager.getInstance().getImage("FIRE_TOWER")));
    			break;
    		case Tower.ICE:
    			tower.AddComponent(new ImageRenderComponent("TowerRender",
    					ResourceManager.getInstance().getImage("ICE_TOWER")));
    			break;
    		}
            tower.AddComponent(new LaserTowerComponent("LaserTower"));
            towerList.add(tower);
        }
        
    }
    
    public void addTower(Tower tower) {
        // Check if player can afford to add the tower
        if(Player.getInstance().getCash()-Player.getInstance().getTowerCost(tower.getType()) >=0) {
            Player.getInstance().addTower(tower.getType());
            towerList.add(tower);
        }
    }

    public void deleteTower(Tower tower) {
        towerList.remove(tower);
    }

    public ArrayList<Tower> getTowers() {
        return towerList;
    }

    
    /*
     * Use to compare against current critter list
     * and then update towerManager's critter list
     * if required
     */
    public ArrayList<Critter> getCritterList() {
        return critterList;
    }

    /*
     * Update the critterList that towers use for finding closest critter
     * Currently called during each update, using a counter to delay it
     * Possible to only update when a critter dies.
     */
    public void updateCritterList(ArrayList<Critter> critterList) {
        this.critterList = critterList;
    }

    /*
     * Returns true if the current position contains a critter
     * Used to stop user from placing towers on critters
     * Doesn't seem to work atm
     */
    public boolean containsCritter(Tower tower) {
        for(Critter critter: critterList) {
            if(critter.getTilePosition() == tower.getTilePosition() ) {
                return true;
            }
        }
        return false;
    }
    

    public void update(GameContainer gc, StateBasedGame sb, int delta) {
        
        for (int i = 0; i < towerList.size(); i++) {
            towerList.get(i).updateCritterList(critterList);
            towerList.get(i).update(gc, sb, delta);
            if (towerList.get(i).isDead()) {
                GameplayState.pathmap.setEmptyTerrain(towerList.get(i).getTilePosition());
                towerList.remove(i);
            }
        }

    }
    
    public void render(GameContainer gc, StateBasedGame sb, Graphics gr) {
        for(Tower tower : towerList) {
            tower.render(gc, sb, gr);
        }
    }




}
