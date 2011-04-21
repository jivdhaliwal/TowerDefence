package towerdefence.engine.entity;

import org.newdawn.slick.GameContainer;
import org.newdawn.slick.state.StateBasedGame;
import towerdefence.engine.component.Component;

/**
 *
 * Critter - An entity that traverses the map avoiding obstacles
 * and tries to reach a goal.
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class Critter extends Entity {


    // Critter types
    public final static int NORMAL = 0;
    public final static int FIRE = 1;
    public final static int ICE = 2;
    public final static int BOSS = 3;
    private int type;

    public Critter(String id) {
        super(id);
        isDead = false;
        health = 100;
    }

    public void takeDamage(float damage) {
        health-=damage;
    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta)
    {

        if(health<=0) {
            isDead = true;
        }

        for(Component component : components)
        {
            component.update(gc,sb,delta);
        }
    }



}
