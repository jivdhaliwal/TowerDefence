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
import org.newdawn.slick.util.pathfinding.PathFinder;

/**
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class CritterFollowPathComponent extends Component {

    PathFinder finder;
    Path path;

    Step currentStep, targetStep;
    private int targetIndex;

    public static final int UP = 0;
    public static final int DOWN = 1;
    public static final int LEFT = 2;
    public static final int RIGHT = 3;

    float distance;
    int moveCounter;
    float critterSpeed;

    private Vector2f position;
    

    public CritterFollowPathComponent( String id, PathFinder finder, Path path ) {
        this.id = id;
        this.finder = finder;
        this.path = path;

        targetIndex = 1;
        distance = 32;

        //Set speed of critters. Anything higher than 0.2f is unstable at 60fps.
        critterSpeed = 0.08f;
              
    }

    /**
     * Tile mover
     * 
     * Move entity from current position to specified tile
     *
     * @param currentPos current position of the entity
     * @param cx current X tile
     * @param cy current Y tile
     * @param tx target X tile
     * @param ty target Y tile
     * 
     */
    public void moveToTile(Vector2f currentPos, Step currentTile, Step targetTile, int delta) {

        int currentX = currentTile.getX();
        int currentY = currentTile.getY();
        int targetX = targetTile.getX();
        int targetY = targetTile.getY();

        entity.setPosition(new Vector2f((position.x + (targetX - currentX)*(critterSpeed*delta)),
                    (position.y + (targetY - currentY)*(critterSpeed*delta))));

        if((targetY - currentY)==1) {
            entity.setDirection(DOWN);
        } else if((targetY - currentY)==-1) {
            entity.setDirection(UP);
        } else if((targetX - currentX)==1) {
            entity.setDirection(RIGHT);
        } else if((targetX - currentX)==-1) {
            entity.setDirection(LEFT);
        }

    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta) {

        position = entity.getPosition();

        if (path!=null && targetIndex < path.getLength()) {
            currentStep = path.getStep(targetIndex - 1);
            targetStep = path.getStep(targetIndex);
            if (distance > 0) {
                moveToTile(position, currentStep, targetStep, delta);
                distance-=delta*critterSpeed;
            } else if (distance <= 0) {
                entity.setPosition(new Vector2f(targetStep.getX()*32,targetStep.getY()*32));
                targetIndex++;
                distance = 32f;
            }
        } else if((path==null || targetIndex >= path.getLength())) {
            entity.killEntity();
        }
    }

}
