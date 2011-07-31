package towerdefence.engine.entity;

import org.newdawn.slick.Color;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.ShapeFill;
import org.newdawn.slick.geom.Rectangle;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.state.StateBasedGame;
import towerdefence.GameplayState;
import towerdefence.engine.Settings;
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
    
    private Rectangle healthBar;
    private ShapeFill healthCol;
    private boolean isSlowed;
    private int slowTimer;

    public Critter(String id) {
        super(id);
        isDead = false;
        health = 100;
        healthBar = new Rectangle(position.x, position.y, GameplayState.TILESIZE, 3);
        isSlowed = false;
    }
    
    /**
     * @return the type
     */
    @Override
    public int getType() {
        return type;
    }

    
    @Override
    public void setType(int type) {
        this.type = type;
        setHealth(Settings.getInstance().getCritterHealth()[type]);
    }

    public void takeDamage(float damage) {
        setHealth(getHealth() - damage);
        healthBar.setSize((getHealth()/Settings.getInstance().getCritterHealth()[type])*GameplayState.TILESIZE, 3);
    }
    
    @Override
    public void setPosition(Vector2f position)
    {
        this.position = position;
        healthBar.setX(position.x);
        healthBar.setY(position.y);
        
    }
    
    // TODO Implement slowed critter
    public void slowCritter(int time) {
        isSlowed = true;
        slowTimer = time;
    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta)
    {

        
        if(getHealth()<=0) {
            isDead = true;
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
        
        gr.setColor(Color.white);
        gr.draw(healthBar);
        gr.setColor(Color.red);
        gr.fill(healthBar);
        gr.setColor(Color.white);
        
    }

}
