package towerdefence.actors;

/**
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */

import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Input;
import org.newdawn.slick.state.StateBasedGame;
import towerdefence.base.Component;

public class TopDownMovement extends Component {

    float direction;
    float speed;

    public TopDownMovement( String id )
    {
        this.id = id;
    }

    public float getSpeed()
    {
        return speed;
    }

    public float getDireciton()
    {
        return direction;
    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta) {

        float rotation = owner.getRotation();
        float scale = owner.getScale();

        Vector2f position = owner.getPosition();

        Input input = gc.getInput();

        if(input.isKeyDown(Input.KEY_DOWN))
        {
            position.x += 1;
        }
        if(input.isKeyDown(Input.KEY_UP))
        {
            position.x -= 1;
        }
        if(input.isKeyDown(Input.KEY_LEFT))
        {
            position.y -= 1;
        }
        if(input.isKeyDown(Input.KEY_RIGHT))
        {
            position.y += 1;
        }

        owner.setPosition(position);

        owner.setRotation(rotation);

        owner.setScale(scale);

    }

}
