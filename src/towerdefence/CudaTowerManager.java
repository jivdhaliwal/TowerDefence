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
public class CudaTowerManager {

    private ArrayList<CudaTower> towerList = new ArrayList<CudaTower>();
    private ArrayList<Critter> critterList;
    private int[] towerArray= new int[0];
    private int[] critterArray= new int[0];
    private int[] targetCritters=null;
    
    private final Image towerSprite;

    CudaCritterSelector cudaSelecter;

    
    
    public CudaTowerManager() throws SlickException {

        towerSprite = new Image("data/sprites/towers/greentower.png");
        
        cudaSelecter = new CudaCritterSelector();

    }

    /*
     * Default Tower when no type is defined
     */
    public void addTower(String id, Vector2f position) throws SlickException {
        CudaTower tower = new CudaTower(id);
        tower.setPosition(position);
        tower.AddComponent(new ImageRenderComponent("CritterRender", towerSprite));
        towerList.add(tower);
        generateTowerArray();
    }

    private void generateTowerArray() {
        towerArray = new int[towerList.size()*2];
        for(int i=0; i<towerList.size(); i++) {
            towerArray[i*2] = (int)towerList.get(i).getPosition().x;
            towerArray[(i*2)+1] = (int)towerList.get(i).getPosition().y;
        }
    }

    public void deleteTower(CudaTower tower) {
        towerList.remove(tower);
    }

    public ArrayList<CudaTower> getTowers() {
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
    
    private void generateCritterArray() {
        critterArray = new int[critterList.size() * 2];
        for (int i = 0; i < critterList.size(); i++) {
            critterArray[i * 2] = (int) critterList.get(i).getPosition().x;
            critterArray[(i * 2) + 1] = (int) critterList.get(i).getPosition().y;
        }
    }


    public void update(GameContainer gc, StateBasedGame sb, int delta) {
        
        if(critterList.size()>0) {
            if(towerList.size()>0) {
                generateCritterArray();
                targetCritters = cudaSelecter.selectCritters(critterArray, towerArray, 128);
                for(int j=0;j<targetCritters.length;j++) {
                    if(targetCritters[j]!=-1) {
                        towerList.get(j).setTargetCritter(critterList.get(targetCritters[j]));
                    } else {
                        towerList.get(j).setTargetCritter(null);
                    }
                }
            }
        }
        
        
        for (CudaTower tower : towerList) {
            tower.update(gc, sb, delta);
        }
    }
    
    public void render(GameContainer gc, StateBasedGame sb, Graphics gr) {
        for(CudaTower tower : towerList) {
            tower.render(gc, sb, gr);
        }
    }




}
