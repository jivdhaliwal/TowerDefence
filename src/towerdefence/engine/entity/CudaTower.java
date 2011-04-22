package towerdefence.engine.entity;

import java.util.ArrayList;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.state.StateBasedGame;
import towerdefence.engine.component.Component;

/**
 *
 * Tower - An entity that deals with distance checking against critters
 * and shoots critters
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class CudaTower extends Entity {

    ArrayList<Critter> critterList = null;
    Critter targetCritter = null;

    private final Image yellowLaser;
    private final Image blueLaser;
    private final Image greenLaser;
    private final Image purpleLaser;
    private final Image redLaser;
    private Image laser;

    float range;
    float damagePerSec;

    private int shootingCounter;

    boolean isShooting;


    public CudaTower(String id) throws SlickException{
        super(id);
        range = 128;
        damagePerSec = 25;

        this.rotation=0;

        yellowLaser = new Image("data/sprites/laser/yellow.png");
        blueLaser = new Image("data/sprites/laser/blue.png");
        greenLaser = new Image("data/sprites/laser/green.png");
        purpleLaser = new Image("data/sprites/laser/purple.png");
        redLaser = new Image("data/sprites/laser/red.png");
        laser = greenLaser;
    }

    public void setTargetCritter(Critter critter) {
        this.targetCritter = critter;
    }
    
    private void shootCritter(Critter critter) {
        critter.takeDamage(damagePerSec/10f);
    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta)
    {
        shootingCounter-=delta;

        if(shootingCounter<=0) {
            if(targetCritter!=null) {
                shootCritter(targetCritter);
            }
            shootingCounter=100;
        }

        for(Component component : components)
        {
            component.update(gc,sb,delta);
        }

    }

    @Override
    public void render(GameContainer gc, StateBasedGame sb, Graphics gr)
    {

        if(renderComponent != null) {
            renderComponent.render(gc, sb, gr);
        }
        // Laser shooting
        // Check tower has a target
        if (targetCritter != null) {

            gr.rotate(this.getPosition().x + 16, this.getPosition().y + 16,
                    (float) (targetCritter.getPosition().sub(this.getPosition())).getTheta()-90);
            // Draw lazer and extend it using the distance from the tower to the
            // target critter
            laser.draw(this.getPosition().x, this.getPosition().y+16, 32,
                    this.getPosition().distance(targetCritter.getPosition()));
            gr.rotate(this.getPosition().x + 16, this.getPosition().y + 16,
                    (float) -(targetCritter.getPosition().sub(this.getPosition())).getTheta()+90);
        }

    }
    
}
