package towerdefence.engine.component;

/**
 *
 * This will help with clicking and placing towers when implemented
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */

import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import javax.swing.event.MouseInputListener;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Input;
import org.newdawn.slick.state.StateBasedGame;

public class MouseMovement extends Component implements MouseMotionListener {

    public MouseMovement( String id )
    {
        this.id = id;
    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public void mouseDragged(MouseEvent e) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public void mouseMoved(MouseEvent e) {
        throw new UnsupportedOperationException("Not supported yet.");
    }


}
