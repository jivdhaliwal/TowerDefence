package towerdefence.engine.component;


import java.util.logging.Level;
import java.util.logging.Logger;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.state.StateBasedGame;

/**
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class ImageRenderComponent extends RenderComponent {

    Image image;

    public ImageRenderComponent(String id, Image image)
    {
        super(id);
        this.image = image;
    }

    @Override
    public void render(GameContainer gc, StateBasedGame sb, Graphics gr) 
    {
        Vector2f pos = entity.getPosition();
        float scale = entity.getScale();
//        try {
//            image.setFilter(Image.FILTER_NEAREST);
//        } catch (SlickException ex) {
//            Logger.getLogger(ImageRenderComponent.class.getName()).log(Level.SEVERE, null, ex);
//        }
        image.draw(pos.x, pos.y, scale);
    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta) 
    {
        image.rotate(entity.getRotation() - image.getRotation());
    }

}
