package towerdefence.engine.component;

import java.util.ArrayList;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.state.StateBasedGame;
import towerdefence.engine.entity.Critter;
import towerdefence.engine.entity.Tower;

/**
 *
 * Handles shooting critters
 *
 * Each tower has a ArrayList of all Critters alive
 *
 * For each critter
 *   Do distance checks
 *     If distance < tower's range
 *        Shoot critter
 *
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class TowerComponent extends Component{

    ArrayList<Critter> critterList;
    
    // The critter in range that the tower has locked onto
    Tower targetCritter;
    private int updateCounter;

    public TowerComponent(String id) {
        this.id = id;
    }

    public void findClosestCritter() {

        for (Critter enemy : critterList) {

            if(entity.getPosition().distanceSquared(enemy.getPosition()) < 30f) {
                shoot( enemy, 30f);
            }

        }

    }

    public void shoot( Critter enemy, float damage) {
    }

    /*
     * Used to update the critter list for each tower to distance check against
     */
    public void setCritterList(ArrayList<Critter> critterList) {
        this.critterList = critterList;
    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta){

        updateCounter-=delta;
        
        if(updateCounter<=0) {
            findClosestCritter();
            updateCounter = 100;
        }
    }

}
