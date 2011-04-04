package towerdefence.engine.entity;

import java.util.ArrayList;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.state.StateBasedGame;
import towerdefence.engine.component.Component;

/**
 *
 * Tower - An entity that deals with distance checking against critters
 * and shoots critters
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class Tower extends Entity {

    ArrayList<Critter> critterList = null;
    Critter targetCritter = null;

    float range;
    float baseDamage;

    int shootingCounter;

    boolean isShooting;

    public Tower(String id){
        super(id);
        range = 200;
        baseDamage = 10;
    }

    /*
     * Update the list of critters the tower iterates through to find a target
     */
    public void updateCritterList(ArrayList<Critter> critterList) {
        this.critterList = critterList;
    }

    private void findClosestCritter() {
        if(critterList!=null) {


            if(targetCritter==null) {
                for(Critter enemy : critterList) {
                    if(this.getPosition().distance(enemy.getPosition()) < range) {
                        targetCritter=enemy;
                        break;
                    }
                }
            } else if (targetCritter!=null) {
                if(this.getPosition().distance(targetCritter.getPosition()) >= range) {
                    targetCritter = null;
                } else if(targetCritter.isDead()) {
                    targetCritter=null;
                } else {
                    shootCritter(targetCritter);
                }
            }
        }
    }

    private void shootCritter(Critter critter) {
        critter.takeDamage(baseDamage*1.5f);
    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta)
    {
        shootingCounter-=delta;

        if(shootingCounter<=0) {
            findClosestCritter();
            shootingCounter=100;
        }

        for(Component component : components)
        {
            component.update(gc,sb,delta);
        }
    }
    

}
