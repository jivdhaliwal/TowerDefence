/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package towerdefence;

import cuda.CudaCritterSelector;
import java.util.ArrayList;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.state.StateBasedGame;
import towerdefence.engine.AnimationLoader;
import towerdefence.engine.Player;
import towerdefence.engine.component.ImageRenderComponent;
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
    private final Image[][] towerSprites;
    private AnimationLoader spriteLoader = new AnimationLoader();
    
    // Arrays for CUDA
    private int[] towerArray= new int[0];
    private int[] critterArray= new int[0];
    private int[] targetCritters=null;
    CudaCritterSelector cudaSelecter;
    
    // Tower types
    public final static int NORMAL = 0;
    public final static int FIRE = 1;
    public final static int ICE = 2;
    public static boolean cudaTowersEnabled;


    public TowerManager(Image[][] towerSprites) throws SlickException {
        
        this.towerSprites = towerSprites;
        
        cudaSelecter = new CudaCritterSelector();
        
        cudaTowersEnabled=false;
        
    }

    /*
     * Default Tower when no type is defined
     */
    public void addTower(String id, Vector2f position, int type) throws SlickException {
        
        if(Player.getInstance().getCash()-Player.getInstance().getTowerCost(type) >=0) {
            Player.getInstance().addTower(type);
            Tower tower = new Tower(id, true);
            tower.setPosition(position);
            tower.setType(type);
            tower.setSprites(getTowerSprites()[type]);
            tower.AddComponent(new ImageRenderComponent("TowerRender", getTowerSprites()[type][0]));
            towerList.add(tower);
            generateTowerArray();
        }
        
    }
    
    public void addTower(Tower tower) {
        if(Player.getInstance().getCash()-Player.getInstance().getTowerCost(tower.getType()) >=0) {
            Player.getInstance().addTower(tower.getType());
            towerList.add(tower);
            generateTowerArray();
        }
    }

    public void deleteTower(Tower tower) {
        towerList.remove(tower);
    }

    public ArrayList<Tower> getTowers() {
        return towerList;
    }

    private void generateTowerArray() {
        towerArray = new int[towerList.size() * 2];
        for (int i = 0; i < towerList.size(); i++) {
            towerArray[i * 2] = (int) towerList.get(i).getPosition().x;
            towerArray[(i * 2) + 1] = (int) towerList.get(i).getPosition().y;
        }
    }
    
    private void generateCritterArray() {
        critterArray = new int[critterList.size() * 2];
        for (int i = 0; i < critterList.size(); i++) {
            critterArray[i * 2] = (int) critterList.get(i).getPosition().x;
            critterArray[(i * 2) + 1] = (int) critterList.get(i).getPosition().y;
        }
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
    
    /**
     * @return the cudaTowersEnabled
     */
    public boolean isCudaTowersEnabled() {
        return cudaTowersEnabled;
    }

    /**
     * @param aCudaTowersEnabled the cudaTowersEnabled to set
     */
    public void setCudaTowersEnabled(boolean aCudaTowersEnabled) {
        this.cudaTowersEnabled = aCudaTowersEnabled;
    }
    
    
    public void update(GameContainer gc, StateBasedGame sb, int delta) {
        
        if(isCudaTowersEnabled()) {
            if (critterList.size() > 0) {
                if (towerList.size() > 0) {
                    generateCritterArray();
                    targetCritters = cudaSelecter.selectCritters(critterArray, towerArray, (int) 128 * 128);
                    for (int j = 0; j < targetCritters.length; j++) {
                        if (targetCritters[j] != -1) {
                            towerList.get(j).setTargetCritter(critterList.get(targetCritters[j]));
                        } else {
                            towerList.get(j).setTargetCritter(null);
                        }
                    }
                }
            }
        }
        
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

    /**
     * @return the towerSprites
     */
    public Image[][] getTowerSprites() {
        return towerSprites;
    }




}
