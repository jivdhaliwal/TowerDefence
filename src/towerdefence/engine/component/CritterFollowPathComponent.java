/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package towerdefence.engine.component;

import org.newdawn.slick.GameContainer;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.state.StateBasedGame;
import org.newdawn.slick.util.pathfinding.Path;
import org.newdawn.slick.util.pathfinding.Path.Step;

/**
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class CritterFollowPathComponent extends Component {

    Path path;

    Step currentStep, targetStep;



    int currentIndex;

    int targetX, targetY;

    int direction;
    
    private static final int STOP = 0;
    private static final int UP = 1;
    private static final int DOWN = 2;
    private static final int LEFT = 3;
    private static final int RIGHT = 4;

    Vector2f currentTile;
    Vector2f previousTile;

    public CritterFollowPathComponent( String id, Path path ) {
        this.id = id;
        this.path = path;
        currentIndex = 0;
        
        currentStep = path.getStep(currentIndex);
        targetStep = path.getStep(currentIndex+1);
        targetX = targetStep.getX();
        targetY = targetStep.getY();

        direction =  UP;

    }
    /*
    public void moveCritter(int direction, int speed) {
        int distance = 0;
        if(direction==LEFT) {
            while(distance<32) {
                entity.setPosition(new Vector2f(position.x += 0.1f*delta,position.y));
            }
        }
    }

     * 
     */
    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta) {

        Vector2f position = entity.getPosition();
        Vector2f tilePosition = entity.getTilePosition(32);

        currentStep = path.getStep(currentIndex);
        targetStep = path.getStep(currentIndex+1);

        targetX = targetStep.getX();
        targetY = targetStep.getY();

        if(targetX - tilePosition.x == 1) {
            entity.setPosition(new Vector2f(position.x += 0.1f*delta,position.y));
        } else if(targetX - tilePosition.x == -1) {
            entity.setPosition(new Vector2f(position.x -= 0.1f*delta,position.y));
        } else if(targetY - tilePosition.y == 1) {
            entity.setPosition(new Vector2f(position.x,position.y += 0.1f*delta));
        } else if(targetY - tilePosition.y == -1) {
            entity.setPosition(new Vector2f(position.x,position.y -= 0.1f*delta));
        } else if(targetX - tilePosition.x == 0 && targetY - tilePosition.y == 0) {
            currentIndex++;
        }

    }

}
