package towerdefence.engine.entity;

import org.newdawn.slick.Color;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.geom.Rectangle;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.state.StateBasedGame;
import towerdefence.GameplayState;
import towerdefence.engine.Settings;
import towerdefence.engine.component.Component;
import towerdefence.engine.component.RenderComponent;


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
    
    public static final int UP = 0;
    public static final int DOWN = 1;
    public static final int LEFT = 2;
    public static final int RIGHT = 3;
    int direction;
    
    private float health;
    
    private Rectangle healthBar;
    
    private Rectangle collisionBlock;

    public Critter(String id) {
        super(id);
        isDead = false;
        health = 100;
        healthBar = new Rectangle(position.x, position.y, GameplayState.TILESIZE, 3);
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
        if(collisionBlock!=null){
        	collisionBlock.setLocation(position);
        }
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
    	if(renderComponents != null) {
            for(RenderComponent sRenderComponent : renderComponents) {
            	sRenderComponent.render(gc, sb, gr);
            }
        }
        
        gr.setColor(Color.white);
        gr.draw(healthBar);
        gr.setColor(Color.red);
        gr.fill(healthBar);
        gr.setColor(Color.white);
        
//        gr.draw(collisionBlock);
        
    }

	public int getDirection() {
		return direction;
	}

	public void setDirection(int direction) {
		this.direction = direction;
	}

	public float getHealth() {
		return health;
	}

	public void setHealth(float health) {
		this.health = health;
	}

	public Rectangle getCollisionBlock() {
		return collisionBlock;
	}

	public void setCollisionBlock(Rectangle collisionBlock) {
		this.collisionBlock = collisionBlock;
	}

}
