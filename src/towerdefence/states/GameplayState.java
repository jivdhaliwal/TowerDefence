/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package towerdefence.states;

import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.state.BasicGameState;
import org.newdawn.slick.state.StateBasedGame;
import org.newdawn.slick.tiled.TiledMap;
import towerdefence.engine.component.TopDownMovement;

import towerdefence.engine.entity.Entity;
import towerdefence.engine.component.ImageRenderComponent;

/**
 *
 * @author Jiv
 */
public class GameplayState extends BasicGameState {

    int stateID = -1;

    private boolean[][] blocked;
    private static final int TILESIZE = 32;

    private TiledMap map;
    private Image testerSprite;

    private float x,y;

    Entity critter = null;

    GameplayState(int stateID) {
        this.stateID = stateID;
    }

    @Override
    public int getID() {
        return 1;
    }

    public void init(GameContainer container, StateBasedGame game) throws SlickException {

        testerSprite = new Image("data/sprites/tester.png");
        map = new TiledMap("data/maps/map1.tmx");

        critter = new Entity("critter");
        critter.AddComponent(new ImageRenderComponent("CritterRender", testerSprite));
        critter.AddComponent(new TopDownMovement("CritterMovement"));
        critter.setPosition(new Vector2f(300, 300));

        // build a collision map based on tile properties in the TileD map
        blocked = new boolean[map.getWidth()][map.getHeight()];
        for (int xAxis = 0; xAxis < map.getWidth(); xAxis++) {
            for (int yAxis = 0; yAxis < map.getHeight(); yAxis++) {
                int tileID = map.getTileId(xAxis, yAxis, 0);

                String value = map.getTileProperty(tileID, "blocked", "false");
                if ("true".equals(value)) {
                    blocked[xAxis][yAxis] = true;
                }
            }
        }

    }

    public void render(GameContainer container, StateBasedGame game, Graphics g) throws SlickException {
        map.render(0,0,1,1,18,18);
        critter.render(container, game, g);
    }

    public void update(GameContainer container, StateBasedGame game, int delta) throws SlickException {

        critter.update(container, game, delta);

    }

    private boolean isBlocked(float x, float y) {
        int xTile = (int) Math.floor((x / TILESIZE));
        int yTile = (int) Math.floor((y / TILESIZE));
        
        return blocked[xTile+1][yTile+1];
    }

}
