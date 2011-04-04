package towerdefence.engine.entity;

import java.util.ArrayList;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.SlickException;
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

    Image laser;

    float range;
    float damagePerSec;

    int shootingCounter;

    boolean isShooting;

    public Tower(String id) throws SlickException{
        super(id);
        range = 200;
        damagePerSec = 10;

        this.rotation=0;

        laser = new Image("data/sprites/laser/blue.png");
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
        critter.takeDamage(damagePerSec/10);
    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta)
    {
        shootingCounter-=delta;

        if(shootingCounter<=0) {
            findClosestCritter();
            shootingCounter=10;
        }

        for(Component component : components)
        {
            component.update(gc,sb,delta);
        }
    }

    @Override
    public void render(GameContainer gc, StateBasedGame sb, Graphics gr)
    {

        // Laser shooting
        // Check tower has a target
        if (targetCritter != null) {
            gr.rotate(this.getPosition().x + 16, this.getPosition().y + 16,
                    (float) (targetCritter.getPosition().sub(this.getPosition())).getTheta()-90);
            // Draw lazer and extend it using the distance from the tower to the
            // target critter
            laser.draw(this.getPosition().x, this.getPosition().y, 32,
                    this.getPosition().distance(targetCritter.getPosition()));
            gr.rotate(this.getPosition().x + 16, this.getPosition().y + 16,
                    (float) -(targetCritter.getPosition().sub(this.getPosition())).getTheta()+90);
        }

        if(renderComponent != null)
            renderComponent.render(gc, sb, gr);


    }
    

}
