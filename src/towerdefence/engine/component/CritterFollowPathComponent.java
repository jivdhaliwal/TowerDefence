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
    private int xDirection;
    private int yDirection;
    private Vector2f position;
    private Vector2f tilePosition;
    private int previousX;
    private int previousY;

    public CritterFollowPathComponent( String id, PathFinder finder, Path path ) {
        this.id = id;
        this.finder = finder;
        this.path = path;
        
        currentIndex = 0;
        
        currentStep = path.getStep(0);
        targetStep = path.getStep(1);

        targetX = targetStep.getX();
        targetY = targetStep.getY();

        
        //xDirection = LEFT;
        //yDirection = UP;
        
    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta) {
        
        if(path!=null){

            position = entity.getPosition();
            tilePosition = entity.getTilePosition(32);

            currentStep = path.getStep(0);
            targetStep = path.getStep(1);

            previousX = currentStep.getX();
            previousY = currentStep.getY();
            targetX = targetStep.getX();
            targetY = targetStep.getY();

            //Vector2f targetVec = new Vector2f((targetX * 32) - 16, (targetY * 32) - 16);
            path = finder.findPath(new UnitMover(3), (int) entity.getTilePosition(32).x,
                    (int) entity.getTilePosition(32).y, 1, 1);


            entity.setPosition(new Vector2f((position.x + ((0.1f * delta) * (targetX - previousX))),
                    (position.y + ((0.1f * delta) * (targetY - previousY)))));
        }
    }

}
