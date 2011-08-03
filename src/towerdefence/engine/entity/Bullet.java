package towerdefence.engine.entity;

import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.geom.Rectangle;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.state.StateBasedGame;

import towerdefence.engine.component.RenderComponent;

public class Bullet extends Entity {

	private Critter targetCritter;
	
	private Rectangle collisionBlock;

	public Bullet(String id) {
		super(id);
	}

	public Critter getTargetCritter() {
		return targetCritter;
	}

	public void setTargetCritter(Critter targetCritter) {
		this.targetCritter = targetCritter;
	}

	public Rectangle getCollisionBlock() {
		return collisionBlock;
	}

	public void setCollisionBlock(Rectangle collisionBlock) {
		this.collisionBlock = collisionBlock;
	}
	
	@Override
	public void setPosition(Vector2f position)
    {
        this.position = position;
        if(collisionBlock!=null) {
        	collisionBlock.setLocation(position);
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
        
//        gr.draw(collisionBlock);
    }

}
