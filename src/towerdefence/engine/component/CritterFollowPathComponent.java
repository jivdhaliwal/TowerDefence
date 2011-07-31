package towerdefence.engine.component;

import org.newdawn.slick.GameContainer;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.state.StateBasedGame;
import org.newdawn.slick.util.pathfinding.Path;
import org.newdawn.slick.util.pathfinding.Path.Step;
import towerdefence.GameplayState;
import towerdefence.engine.Player;
import towerdefence.engine.Settings;

/**
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class CritterFollowPathComponent extends Component {

    Step currentStep, targetStep;
    private int targetIndex;

    // Critter types
    public final static int NORMAL = 0;
    public final static int FIRE = 1;
    public final static int ICE = 2;
    public final static int BOSS = 3;

    public static final int UP = 0;
    public static final int DOWN = 1;
    public static final int LEFT = 2;
    public static final int RIGHT = 3;

    float distance;
    int moveCounter;
    float critterSpeed;

    private Vector2f position;

    private Path path;
    

    public CritterFollowPathComponent( String id , Path path, int type) {
        this.id = id;

        targetIndex = 1;
        distance = GameplayState.TILESIZE;

        critterSpeed= (float)Settings.getInstance().getCritterSpeed()[type] * 0.08f;
        
        
        this.path = path;
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

        if(entity!=null) {
            position = entity.getPosition();
        

            if (path != null && targetIndex < path.getLength()) {
                currentStep = path.getStep(targetIndex - 1);
                targetStep = path.getStep(targetIndex);
                if (distance > 0) {
                    moveToTile(position, currentStep, targetStep, delta);
                    distance -= delta * critterSpeed;
                } else if (distance <= 0) {
                    entity.setPosition(new Vector2f(targetStep.getX() * GameplayState.TILESIZE,
                            targetStep.getY() * GameplayState.TILESIZE));
                    targetIndex++;
                    distance = (float) GameplayState.TILESIZE;
                }
            } else if (targetIndex >= path.getLength()) {
                Player.getInstance().setHealth(Player.getInstance().getHealth() - 1);
                entity.deleteEntity();
            }
        }
    }

}
