package towerdefence.engine.component;


import org.newdawn.slick.Animation;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.state.StateBasedGame;

import towerdefence.engine.entity.Critter;
import towerdefence.engine.entity.Entity;

/**
 *
 * Eventually this will manage rendering animations by loading them from sprite sheets
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class CritterAnimationComponent extends RenderComponent {

    private Animation sprite, up,down,left,right;
    
    private Critter critter;

    public CritterAnimationComponent(String id, Animation[] animation) throws SlickException
    {
        super(id);

        up = animation[0];
        down = animation[1];
        left = animation[2];
        right = animation[3];

        sprite = left;

    }
    
    @Override
    public void setOwnerEntity(Entity entity)
    {
        this.entity = entity;
        critter = (Critter)entity;
    }

    @Override
    public void render(GameContainer gc, StateBasedGame sb, Graphics gr) {
        // Using 64x64 critters (looks better) on a 32x32 grid so sprites need to
        // be shifted left 16pixels and up 32 pixels to allign correctly.
        sprite.draw(critter.getPosition().x-16, critter.getPosition().y-32);
    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta) {

        if (critter.getDirection() == Critter.LEFT) {
            sprite = left;
        }
        if (critter.getDirection() == Critter.RIGHT) {
            sprite = right;
        }
        if (critter.getDirection() == Critter.UP) {
            sprite = up;
        }
        if (critter.getDirection() == Critter.DOWN) {
            sprite = down;
        }

         
    }


}
