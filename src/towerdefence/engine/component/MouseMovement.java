package towerdefence.engine.component;

/**
 *
 * This will help with clicking and placing towers when implemented
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */

import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Input;
import org.newdawn.slick.MouseListener;
import org.newdawn.slick.state.StateBasedGame;

public class MouseMovement extends Component {

    public MouseMovement( String id )
    {
        this.id = id;
    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta) {
        Input i = gc.getInput();i.addMouseListener(new MouseListener(){

            public void mouseWheelMoved(int change) {
            }

            public void mouseClicked(int button, int x, int y, int clickCount) {
            }

            public void mousePressed(int button, int x, int y) {
            }

            public void mouseReleased(int button, int x, int y) {
            }

            public void mouseMoved(int oldx, int oldy, int newx, int newy) {
            }

            public void mouseDragged(int oldx, int oldy, int newx, int newy) {
            }

            public void setInput(Input input) {
            }

            public boolean isAcceptingInput() {
                return true;
            }

            public void inputEnded() {
            }

            public void inputStarted() {
            }
        });
        
    }




}
