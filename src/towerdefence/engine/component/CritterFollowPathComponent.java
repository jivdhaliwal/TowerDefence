package towerdefence.engine.component;

import org.newdawn.slick.GameContainer;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.state.StateBasedGame;
import org.newdawn.slick.util.pathfinding.Path;
import org.newdawn.slick.util.pathfinding.Path.Step;
import towerdefence.GameplayState;
import towerdefence.engine.Player;
import towerdefence.engine.Settings;
import towerdefence.engine.entity.Critter;
import towerdefence.engine.entity.Entity;

/**
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class CritterFollowPathComponent extends Component {

    Step currentStep, targetStep;
    private int targetIndex;

    float distance;
    int moveCounter;
    float critterSpeed;

    private Vector2f position;

    private Path path;
    
    private Critter critter;

    public CritterFollowPathComponent( String id , Path path, int type) {
        this.id = id;

        targetIndex = 1;
        distance = GameplayState.TILESIZE;

        critterSpeed= (float)Settings.getInstance().getCritterSpeed()[type] * 0.08f;
        
        
        this.path = path;
    }
    
    @Override
    public void setOwnerEntity(Entity entity)
    {
        this.entity = entity;
        critter = (Critter)entity;
    }

    /**
     * Tile mover
     * 
     * Move entity from current position to specified tile
     *
     * @param currentPos current position of the entity
     * @param currentTile current tile the critter is walking from
     * @param targetTile target tile the critter is walking towards
     * 
     */
    public void moveToTile(Vector2f currentPos, Step currentTile, Step targetTile, int delta) {

        int currentX = currentTile.getX();
        int currentY = currentTile.getY();
        int targetX = targetTile.getX();
        int targetY = targetTile.getY();

        critter.setPosition(new Vector2f((position.x + (targetX - currentX)*(critterSpeed*delta)),
                    (position.y + (targetY - currentY)*(critterSpeed*delta))));

        if((targetY - currentY)==1) {
            critter.setDirection(Critter.DOWN);
        } else if((targetY - currentY)==-1) {
            critter.setDirection(Critter.UP);
        } else if((targetX - currentX)==1) {
            critter.setDirection(Critter.RIGHT);
        } else if((targetX - currentX)==-1) {
            critter.setDirection(Critter.LEFT);
        }

    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta) {

        if(critter!=null) {
            position = critter.getPosition();
        

            if (path != null && targetIndex < path.getLength()) {
                currentStep = path.getStep(targetIndex - 1);
                targetStep = path.getStep(targetIndex);
                if (distance > 0) {
                    moveToTile(position, currentStep, targetStep, delta);
                    distance -= delta * critterSpeed;
                } else if (distance <= 0) {
                    critter.setPosition(new Vector2f(targetStep.getX() * GameplayState.TILESIZE,
                            targetStep.getY() * GameplayState.TILESIZE));
                    targetIndex++;
                    distance = (float) GameplayState.TILESIZE;
                }
            } else if (targetIndex >= path.getLength()) {
                Player.getInstance().setHealth(Player.getInstance().getHealth() - 1);
                critter.deleteEntity();
            }
        }
    }

}
