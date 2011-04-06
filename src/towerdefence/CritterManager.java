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
import towerdefence.engine.component.CritterAnimationComponent;
import towerdefence.engine.component.CritterFollowPathComponent;
import towerdefence.engine.component.ImageRenderComponent;
import towerdefence.engine.component.TopDownMovement;
import towerdefence.engine.entity.Critter;
import towerdefence.engine.pathfinding.UnitMover;

/**
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */



public class CritterManager {

    int numCritters;

    // Critter types
    public final static int NORMAL = 0;
    public final static int LEVEL1 = 1;
    public final static int LEVEL2 = 2;
    public final static int BOSS = 3;

    Vector2f initialPos;
    Image testerSprite;
    PathFinder finder;
    Path path;

    ArrayList<Critter> critterList = new ArrayList<Critter>();
    private int critterCount;
    private int generateCounter;
    private final int critterType;

    public CritterManager(Vector2f initialPos, PathFinder finder, int critterCount, int critterType) throws SlickException {
        this.initialPos = initialPos;
        this.finder = finder;
        this.path = finder.findPath(new UnitMover(3), getTilePosition(initialPos.x,32),
                getTilePosition(initialPos.y,32), 1,1);
        this.critterCount = critterCount;
        this.critterType = critterType;

        testerSprite = new Image("data/sprites/positionTester.png");


        
    }

    /*
     * Add's critter to ArrayList of critters
     */
    public void addCritter(String id) {
        Critter critter = new Critter(id);
        critter.setPosition(initialPos);

        critter.AddComponent(new ImageRenderComponent("CritterRender", testerSprite));
        critter.AddComponent(new TopDownMovement("CritterMovement"));
        critter.AddComponent(new CritterFollowPathComponent("CritterPath", finder, path));

        critterList.add(critter);
    }

    /*
     * Add's critter to ArrayList of critters
     */
    public void addCritter(String id, boolean Animmated) throws SlickException {
        Critter critter = new Critter(id);
        critter.setPosition(initialPos);

        critter.AddComponent(new CritterAnimationComponent(id, critterType));
        critter.AddComponent(new CritterFollowPathComponent("CritterPath", finder, path));

        critterList.add(critter);
    }

    public void generateCritters(int numCritters) {
        for(int i=0;i<numCritters;i++) {
            addCritter(String.valueOf(i));
        }
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

        // Critter wave generator
        if (generateCounter < 0) {
            if (critterCount > 0) {

                addCritter(String.valueOf(critterCount + 100),true);
                critterCount--;
            }
            generateCounter = 1000;
        }

    }

    public void render(GameContainer container, StateBasedGame game, Graphics g) throws SlickException {

        for (Critter enemy : critterList) {
            enemy.render(container, game, g);
        }

    }

}
