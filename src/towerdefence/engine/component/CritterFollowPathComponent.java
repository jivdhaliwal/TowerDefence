/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package towerdefence.engine.component;

import javax.media.j3d.Shape3D;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.state.StateBasedGame;
import org.newdawn.slick.util.pathfinding.Path;
import org.newdawn.slick.util.pathfinding.Path.Step;
import org.newdawn.slick.util.pathfinding.PathFinder;
import towerdefence.engine.pathfinding.UnitMover;

/**
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class CritterFollowPathComponent extends Component {

    PathFinder finder;
    Path path;

    Step currentStep, targetStep;

    int distance;
    int moveCounter;
    int speed;
    
    private static final int STOP = 0;
    private static final int UP = 1;
    private static final int DOWN = 2;
    private static final int LEFT = 3;
    private static final int RIGHT = 4;

    private Vector2f position;
    private Vector2f tilePosition;
    private int targetIndex;

    public CritterFollowPathComponent( String id, PathFinder finder, Path path ) {
        this.id = id;
        this.finder = finder;
        this.path = path;

        targetIndex = 1;
        distance = 32;
        speed = 32;
              
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

//        entity.setPosition(new Vector2f((position.x + ((0.1f * delta) * (targetX - currentX))),
//                    (position.y + ((0.1f * delta) * (targetY - currentY)))));
        entity.setPosition(new Vector2f((position.x + (targetX - currentX)),
                    (position.y + (targetY - currentY))));

    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta) {

        position = entity.getPosition();
        tilePosition = entity.getTilePosition(32);

        moveCounter -= delta;

        if (targetIndex < path.getLength()) {
            currentStep = path.getStep(targetIndex - 1);
            targetStep = path.getStep(targetIndex);
            if (moveCounter <= 0) {
                if (distance > 0) {
                    moveToTile(position, currentStep, targetStep, delta);
                    distance--;
                    moveCounter = speed;
                } else if (distance <= 0) {
                    targetIndex++;
                    distance = 32;
                    moveCounter = 0;
                }
                //Vector2f targetVec = new Vector2f((targetX * 32) - 16, (targetY * 32) - 16);
                //            path = finder.findPath(new UnitMover(3), (int) entity.getTilePosition(32).x,
                //                    (int) entity.getTilePosition(32).y, 1, 1);

            }
        }

    }

}
