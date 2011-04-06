/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package towerdefence;

import java.util.ArrayList;
import java.awt.Font;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.newdawn.slick.Animation;
import org.newdawn.slick.Color;
import org.newdawn.slick.TrueTypeFont;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.Input;
import org.newdawn.slick.MouseListener;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.SpriteSheet;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.particles.ParticleSystem;
import org.newdawn.slick.state.BasicGameState;
import org.newdawn.slick.state.StateBasedGame;
import org.newdawn.slick.tiled.TiledMap;
import org.newdawn.slick.util.pathfinding.*;
import towerdefence.engine.component.ImageRenderComponent;
import towerdefence.engine.entity.Critter;
import towerdefence.engine.entity.Tower;

import towerdefence.engine.pathfinding.PathMap;
import towerdefence.engine.pathfinding.UnitMover;

/**
 *
 * @author Jiv
 */
public class GameplayState extends BasicGameState {

    int stateID = -1;
    int critterCount;
    int generateCounter;
    int mouseCounter;
    // Map used to find critter paths
    private PathMap pathmap;
    // Path finder used to search the pathmap
    private PathFinder finder;
    // Gives the last path found for the current unit
    private Path path;
    private TiledMap map;
    private Image pathSprite;
    private Image towerSprite;
    private Animation wanderingNPCAnim;

    ArrayList<CritterManager> critterWaveList = new ArrayList<CritterManager>();
    CritterManager critterWave;
    ArrayList<Critter> critterList;
    ArrayList<Tower> towerList = new ArrayList<Tower>();

    Critter critter = null;
    ParticleSystem ps;
    private TrueTypeFont trueTypeFont;
    private Tower testTower;
    private Tower testTower2;
    private TowerManager towerFactory;


    GameplayState(int stateID) {
        this.stateID = stateID;
    }

    @Override
    public int getID() {
        return 1;
    }

    public void init(GameContainer container, StateBasedGame game) throws SlickException {


        pathSprite = new Image("data/sprites/path.png");
        map = new TiledMap("data/maps/path1_5.tmx");
        
        Image normalSheet = new Image("data/sprites/wandering_trader2.png");
        SpriteSheet critterSheet = new SpriteSheet(normalSheet, 32, 64);
        Image[] wanderingNPC = {critterSheet.getSprite(0, 0),critterSheet.getSprite(1, 0),
            critterSheet.getSprite(2, 0), critterSheet.getSprite(3, 0),
            critterSheet.getSprite(4, 0), critterSheet.getSprite(5, 0)};
        wanderingNPCAnim = new Animation(wanderingNPC, 230,true);
        
        pathmap = new PathMap(map);
        finder = new AStarPathFinder(pathmap, 500, false);
        path = finder.findPath(new UnitMover(3), map.getWidth() - 1, map.getHeight() - 1, 1, 1);

        towerFactory = new TowerManager();


        critterCount = 20000;
        critterWave = new CritterManager(
                new Vector2f((float) (32 * (map.getWidth() - 1)), (float) (32 * (map.getHeight() - 1))),
                finder,critterCount,CritterManager.NORMAL );

        critterList = critterWave.getCritters();

        critterWaveList.add(critterWave);

        Font font = new Font("Verdana", Font.PLAIN, 20);
        trueTypeFont = new TrueTypeFont(font, true);

    }

    public void render(GameContainer container, StateBasedGame game, Graphics g) throws SlickException {
        int tempCritterCount = 0;

        map.render(0, 0);

        // Render the path for testing
        /*
        for (int x = 0; x < map.getWidth(); x++) {
            for (int y = 0; y < map.getHeight(); y++) {
                if (path != null) {
                    if (path.contains(x, y)) {
                        pathSprite.draw(x * 32, y * 32);
                    }
                }
            }
        }
         * 
         */

        for(CritterManager wave : critterWaveList) {
            wave.render(container, game, g);
            tempCritterCount+=wave.getCritters().size();
        }
        towerFactory.render(container, game, g);
        
        trueTypeFont.drawString((map.getWidth() * 32) - 400, 50, "# of Critters : " + String.valueOf(tempCritterCount), Color.white);
        trueTypeFont.drawString((map.getWidth() * 32) - 400, 100, "# of Towers : " + String.valueOf(towerFactory.getTowers().size()), Color.white);

        wanderingNPCAnim.draw(32f,0f);

    }

    public void update(GameContainer container, StateBasedGame game, int delta) throws SlickException {

        ArrayList<Critter> tempCritterList = new ArrayList<Critter>();

        Input input = container.getInput();

        if (input.isKeyPressed(Input.KEY_F2)) {
            game.enterState(TowerDefence.PATHTESTSTATE);
        }
        

        mouseCounter -= delta;

        for(CritterManager wave : critterWaveList) {
            wave.update(container, game, delta);
            // Adds critters from each wave to the temp critter list
            tempCritterList.addAll(wave.getCritters());
        }
        
        towerFactory.updateCritterList(tempCritterList);
        towerFactory.update(container, game, delta);

        input.addMouseListener(new MouseListener(){

            public void mouseWheelMoved(int change) {
            }

            public void mouseClicked(int button, int x, int y, int clickCount) {
                if (mouseCounter <= 0) {
                    int currentXTile = (int) Math.floor((x / 32));
                            int currentYTile = (int) Math.floor((y / 32));
                    if(button==0) {
                        if(pathmap.getTerrain(currentXTile, currentYTile)!=PathMap.GRASS) {
                                try {
                                    towerFactory.addTower(String.valueOf(x), new Vector2f(currentXTile * 32, currentYTile * 32));
                                    mouseCounter=50;
                                } catch (SlickException ex) {
                                    Logger.getLogger(GameplayState.class.getName()).log(Level.SEVERE, null, ex);
                                }
                        }
                        }
                    else if(button==1) {
                        if(pathmap.getTerrain(currentXTile, currentYTile)==PathMap.GRASS) {
                            try {
                                CritterManager newWave = new CritterManager(
                                        new Vector2f(currentXTile * 32, currentYTile * 32),
                                        finder, critterCount, CritterManager.NORMAL);
                                critterWaveList.add(newWave);
                                mouseCounter=50;
                            } catch (SlickException ex) {
                                Logger.getLogger(GameplayState.class.getName()).log(Level.SEVERE, null, ex);
                            }
                            }
                    }
                }
            }

            public void mousePressed(int button, int x, int y) {
            }

            public void mouseReleased(int button, int x, int y) {
            }

            public void mouseMoved(int oldx, int oldy, int newx, int newy) {
            }

            public void mouseDragged(int oldx, int oldy, int newx, int newy) {
            }

            public void setInput(Input input) {
            }

            public boolean isAcceptingInput() {
                return true;
            }

            public void inputEnded() {
            }

            public void inputStarted() {
            }
        });

    }
}
