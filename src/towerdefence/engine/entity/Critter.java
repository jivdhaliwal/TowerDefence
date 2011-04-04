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

    boolean isDead;

    float health;

    public Critter(String id) {
        super(id);
        isDead = false;
        health = 100;
    }

    public float getHealth() {
        return health;
    }

    public void takeDamage(float damage) {
        health-=damage;
    }

    public boolean isDead() {
        return isDead;
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
