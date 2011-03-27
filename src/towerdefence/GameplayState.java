/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package towerdefence;

import java.util.ArrayList;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.geom.Shape;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.state.BasicGameState;
import org.newdawn.slick.state.StateBasedGame;
import org.newdawn.slick.tiled.TiledMap;
import org.newdawn.slick.util.pathfinding.*;
import towerdefence.engine.component.CritterFollowPathComponent;
import towerdefence.engine.component.TopDownMovement;

import towerdefence.engine.entity.Entity;
import towerdefence.engine.component.ImageRenderComponent;
import towerdefence.engine.pathfinding.PathMap;
import towerdefence.engine.pathfinding.UnitMover;

/**
 *
 * @author Jiv
 */
public class GameplayState extends BasicGameState {

    int stateID = -1;

    int critterCount;
    int generateCounter=0;

    private boolean[][] blocked;
    private static final int TILESIZE = 32;

    // Map used to find critter paths
    private PathMap pathmap;
    // Path finder used to search the pathmap
    private PathFinder finder;
    // Gives the last path found for the current unit
    private Path path;


    private TiledMap map;
    private Image testerSprite;
    private Image pathSprite;

    CritterFactory critterFactory;
    ArrayList<Entity> critters = new ArrayList<Entity>();
    Entity critter=null;

    GameplayState(int stateID) {
        this.stateID = stateID;
    }

    @Override
    public int getID() {
        return 1;
    }

    public void init(GameContainer container, StateBasedGame game) throws SlickException {

        testerSprite = new Image("data/sprites/tester.png");
        pathSprite = new Image("data/sprites/path.png");
            map = new TiledMap("data/maps/path1_3.tmx");

        pathmap = new PathMap(map);
        finder = new AStarPathFinder(pathmap, 500, false);
        path = finder.findPath(new UnitMover(3), map.getWidth()-1, map.getHeight()-1, 1, 1);

        /*
        critter = new Entity("critter");
        critter.setPosition(new Vector2f((float) ((32*map.getWidth())-16), (float) ((32*map.getHeight())-16)));
        
        critter.AddComponent(new ImageRenderComponent("CritterRender", testerSprite));
        critter.AddComponent(new TopDownMovement("CritterMovement"));
        critter.AddComponent(nedw CritterFollowPathComponent("CritterPath", finder, path));

        critters.add(critter);
         * 
         */

        critterCount = 1;

        critterFactory = new CritterFactory(
                new Vector2f((float) (32*(21))-16 , (float) (32*(21))-16 ),
                finder);

        //critterFactory.generateCritters(5);

        critters = critterFactory.getCritters();
        
    }

    public void render(GameContainer container, StateBasedGame game, Graphics g) throws SlickException {
        map.render(0,0);
        //map.render(0,0,1,1,18,18);
        
        for (int x = 0; x < map.getWidth(); x++) {
            for (int y = 0; y < map.getHeight(); y++) {
                if(path != null) {
                    if(path.contains(x,y)) {
                        pathSprite.draw(x*32,y*32);
                    }
                }
            }
        }

        for(Entity enemy : critters)
            enemy.render(container, game, g);
    }

    public void update(GameContainer container, StateBasedGame game, int delta) throws SlickException {

        generateCounter-=delta;
        critters = critterFactory.getCritters();

        for(Entity enemy : critters)
            enemy.update(container, game, delta);

        //path = finder.findPath(new UnitMover(3), (int) critter.getTilePosition(32).x, (int) critter.getTilePosition(32).y, 0, 0);

        if(generateCounter < 0) {
            if(critterCount>0){

                critterFactory.addCritter(String.valueOf(critterCount+100));
                critterCount--;
            }
            generateCounter = 500;
        }

    }


    /*
     * Old block checker Might re-implement
     *
     * /
    private boolean isBlocked(float x, float y) {
        int xTile = (int) Math.floor((x / TILESIZE));
        int yTile = (int) Math.floor((y / TILESIZE));
        
        return blocked[xTile+1][yTile+1];
    }
     *
     */

}
