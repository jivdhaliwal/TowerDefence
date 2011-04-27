package towerdefence;

import java.util.ArrayList;
import org.newdawn.slick.Animation;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.state.StateBasedGame;
import org.newdawn.slick.util.pathfinding.Path;
import org.newdawn.slick.util.pathfinding.PathFinder;
import towerdefence.engine.AnimationLoader;
import towerdefence.engine.component.CritterAnimationComponent;
import towerdefence.engine.component.CritterFollowPathComponent;
import towerdefence.engine.component.TopDownMovement;
import towerdefence.engine.entity.Critter;
import towerdefence.engine.pathfinding.UnitMover;

/**
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */



public class CritterManager {

    int generateCounter;

    // Critter types
    public final static int NORMAL = 0;
    public final static int FIRE = 1;
    public final static int ICE = 2;
    public final static int BOSS = 3;

    Vector2f initialPos;
    Image testerSprite;
    PathFinder finder;
    Path path;

    private ArrayList<Critter> critterList = new ArrayList<Critter>();
    private Animation[][] critterAnimation;

    private AnimationLoader animationLoader = new AnimationLoader();
    
    public CritterManager(int startX, int startY, int goalX, int goalY,
            PathFinder finder, Animation[][] critterAnimation) throws SlickException {
        this.finder = finder;
        this.path = finder.findPath(new UnitMover(3), startX, startY, goalX, goalY);
        this.initialPos = new Vector2f(startX * GameplayState.TILESIZE, startY * GameplayState.TILESIZE);
        this.critterAnimation = critterAnimation;

    }


    /*
     * Add's critter to ArrayList of critters
     */
    public void addCritter(String id, int critterType) throws SlickException {
        Critter critter = new Critter(id);
        critter.setPosition(initialPos);
        critter.setType(critterType);

        critter.AddComponent(new CritterAnimationComponent(id, 
                critterAnimation[critterType]));
        critter.AddComponent(new CritterFollowPathComponent("CritterPath", path, critterType));
        critterList.add(critter);
    }

    public ArrayList<Critter> getCritters() {
        return critterList;
    }

    public int getTilePosition(float position, int tilesize)
    {
        return ((int) (Math.floor(position / tilesize)));
    }

    public void update(GameContainer container, StateBasedGame game, int delta) throws SlickException {

        generateCounter-=delta;

        for (Critter enemy : critterList) {
            enemy.update(container, game, delta);
        }

        // Remove dead critters
        for (int i = 0; i < critterList.size(); i++) {
            if (critterList.get(i).isDead()) {
                critterList.remove(i);
            }
        }

    }

    public void render(GameContainer container, StateBasedGame game, Graphics g) throws SlickException {

        for (Critter enemy : critterList) {
            enemy.render(container, game, g);
        }

    }

}
