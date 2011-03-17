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



    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta) {

        Vector2f position = entity.getPosition();
        Vector2f tilePosition = entity.getTilePosition(32);


        System.out.println(tilePosition.y);


        currentStep = path.getStep(currentIndex);
        targetStep = path.getStep(currentIndex+1);


        targetX = targetStep.getX();
        targetY = targetStep.getY();

        

        // Goal Reached. Stop the critter
        if(currentIndex == path.getLength()) {
            direction = STOP;
        }
        else if(targetX - currentStep.getX() == 1) {
            direction = RIGHT;
        }
        else if(targetY - currentStep.getY() == -1) {
            direction = UP;
        }
        else if(targetX - currentStep.getX() == -1) {
            direction = LEFT;
        }
        else if(targetY - currentStep.getY() == 1) {
            direction = DOWN;
        }

        if(direction == RIGHT) {
            position.x += 0.1f*delta;
        }
        if(direction == UP) {
            position.y -= 0.1f*delta;
        }
        if(direction == LEFT) {
            position.x -= 0.1f*delta;
        }
        if(direction == DOWN) {
            position.y += 0.1f*delta;
        }

        if(tilePosition.x==targetStep.getX()) {
            currentIndex++;
        }
        // If critter moves to new tile
        

    }

}
