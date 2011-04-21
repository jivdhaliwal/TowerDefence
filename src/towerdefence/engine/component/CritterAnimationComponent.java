package towerdefence.engine.component;


import org.newdawn.slick.Animation;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.SpriteSheet;
import org.newdawn.slick.state.StateBasedGame;

/**
 *
 * Eventually this will manage rendering animations by loading them from sprite sheets
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class CritterAnimationComponent extends RenderComponent {

    public static final int UP = 0;
    public static final int DOWN = 1;
    public static final int LEFT = 2;
    public static final int RIGHT = 3;

    private Animation sprite, up,down,left,right;

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
    public void render(GameContainer gc, StateBasedGame sb, Graphics gr) {
        // Using 64x64 critters (looks better) on a 32x32 grid so sprites need to
        // be shifted left 16pixels and up 32 pixels to allign correctly.
        sprite.draw(entity.getPosition().x-16, entity.getPosition().y-32);
    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta) {

        if (entity.getDirection() == LEFT) {
            sprite = left;
        }
        if (entity.getDirection() == RIGHT) {
            sprite = right;
        }
        if (entity.getDirection() == UP) {
            sprite = up;
        }
        if (entity.getDirection() == DOWN) {
            sprite = down;
        }

         
    }


}
