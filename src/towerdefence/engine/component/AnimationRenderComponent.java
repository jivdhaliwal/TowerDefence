package towerdefence.engine.component;


import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.state.StateBasedGame;

/**
 *
 * Eventually this will manage rendering animations by loading them from sprite sheets
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class AnimationRenderComponent extends RenderComponent {

    public AnimationRenderComponent(String id, Image image)
    {
        super(id);
    }

    @Override
    public void render(GameContainer gc, StateBasedGame sb, Graphics gr) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta) {
        throw new UnsupportedOperationException("Not supported yet.");
    }


}
