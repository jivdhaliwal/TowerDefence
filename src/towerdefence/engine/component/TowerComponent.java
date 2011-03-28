package towerdefence.engine.component;

import java.util.ArrayList;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.state.StateBasedGame;
import towerdefence.engine.entity.Entity;

/**
 *
 * Handles shooting critters
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class TowerComponent extends Component{

    public TowerComponent(ArrayList<Entity> critterList) {

    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

}
